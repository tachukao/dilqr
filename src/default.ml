open Owl
module AD = Algodiff.D
open AD.Builder

let tmp_dir = Cmdargs.(get_string "-tmp" |> force ~usage:"-tmp [tmp dir]")
let in_tmp_dir = Printf.sprintf "%s/%s" tmp_dir
let () = ignore in_tmp_dir

type t = theta:AD.t -> k:int -> x:AD.t -> u:AD.t -> AD.t
type s = theta:AD.t -> k:int -> x:AD.t -> AD.t
type final_loss = theta:AD.t -> k:int -> x:AD.t -> AD.t
type running_loss = theta:AD.t -> k:int -> x:AD.t -> u:AD.t -> AD.t

let _print_dim str x =
  let shp = AD.Arr.shape x in
  Printf.printf "%s " str;
  Array.iter (Printf.printf "%i  ") shp;
  Printf.printf "\n %!"


let forward_for_backward
    ~theta
    ~dyn_x
    ~dyn_u
    ~rl_uu
    ~rl_xx
    ~rl_ux
    ~rl_u
    ~rl_x
    ~fl_xx
    ~fl_x
    ~dyn
    ()
    x0
    us
  =
  let kf, xf, tape =
    List.fold_left
      (fun (k, x, tape) u ->
        let a = dyn_x ~theta ~k ~x ~u
        and b = dyn_u ~theta ~k ~x ~u
        and rlx = rl_x ~theta ~k ~x ~u
        and rlu = rl_u ~theta ~k ~x ~u
        and rlxx = rl_xx ~theta ~k ~x ~u
        and rluu = rl_uu ~theta ~k ~x ~u
        and rlux = rl_ux ~theta ~k ~x ~u in
        let s = Lqr.{ x; u; a; b; rlx; rlu; rlxx; rluu; rlux } in
        let x = dyn ~theta ~k ~x ~u in
        succ k, x, s :: tape)
      (0, x0, [])
      us
  in
  let flxx = fl_xx ~theta ~k:kf ~x:xf in
  let flx = fl_x ~theta ~k:kf ~x:xf in
  flxx, flx, tape, xf


module type P = sig
  val n : int
  val m : int
  val n_steps : int
  val dyn : t
  val final_loss : final_loss
  val running_loss : running_loss
  val dyn_x : t option
  val dyn_u : t option
  val rl_uu : t option
  val rl_xx : t option
  val rl_ux : t option
  val rl_u : t option
  val rl_x : t option
  val fl_xx : s option
  val fl_x : s option
end

module Make (P : P) = struct
  include P

  let dyn_u =
    let default ~theta ~k ~x ~u =
      AD.jacobian (fun u -> dyn ~theta ~k ~x ~u) u |> AD.Maths.transpose
    in
    Option.value dyn_u ~default


  let dyn_x =
    let default ~theta ~k ~x ~u =
      AD.jacobian (fun x -> dyn ~theta ~k ~x ~u) x |> AD.Maths.transpose
    in
    Option.value dyn_x ~default


  let rl_u =
    let default ~theta ~k ~x ~u = AD.grad (fun u -> running_loss ~theta ~k ~x ~u) u in
    Option.value rl_u ~default


  let rl_x =
    let default ~theta ~k ~x ~u = AD.grad (fun x -> running_loss ~theta ~k ~x ~u) x in
    Option.value rl_x ~default


  let rl_uu =
    let default ~theta ~k ~x ~u =
      AD.jacobian (fun u -> rl_u ~theta ~k ~x ~u) u |> AD.Maths.transpose
    in
    Option.value rl_uu ~default


  let rl_xx =
    let default ~theta ~k ~x ~u =
      AD.jacobian (fun x -> rl_x ~theta ~k ~x ~u) x |> AD.Maths.transpose
    in
    Option.value rl_xx ~default


  let rl_ux =
    let default ~theta ~k ~x ~u = AD.jacobian (fun x -> rl_u ~theta ~k ~x ~u) x in
    Option.value rl_ux ~default


  let fl_x =
    let default ~theta ~k ~x = AD.grad (fun x -> final_loss ~theta ~k ~x) x in
    Option.value fl_x ~default


  let fl_xx =
    let default ~theta ~k ~x =
      AD.jacobian (fun x -> fl_x ~theta ~k ~x) x |> AD.Maths.transpose
    in
    Option.value fl_xx ~default


  let forward ~theta x0 us =
    List.fold_left
      (fun (k, x, xs, us) u ->
        let xs = x :: xs in
        let us = u :: us in
        let x = dyn ~theta ~k ~x ~u in
        succ k, x, xs, us)
      (0, x0, [], [])
      us


  let ffb ~theta =
    forward_for_backward
      ~theta
      ~dyn_x
      ~dyn_u
      ~rl_uu
      ~rl_xx
      ~rl_ux
      ~rl_u
      ~rl_x
      ~fl_xx
      ~fl_x
      ~dyn
      ()


  let update ~theta x0 us =
    (* xf, xs, us are in reverse *)
    let vxxf, vxf, tape, _ = ffb ~theta x0 us in
    let acc, (df1, df2) = Lqr.backward vxxf vxf tape in
    fun alpha ->
      let _, _, uhats =
        List.fold_left
          (fun (k, xhat, uhats) (x, u, (_K, _k)) ->
            let dx = AD.Maths.(xhat - x) in
            let du = AD.Maths.((dx *@ _K) + (AD.F alpha * _k)) in
            let uhat = AD.Maths.(u + du) in
            let uhats = uhat :: uhats in
            let xhat = dyn ~theta ~k ~x:xhat ~u:uhat in
            succ k, xhat, uhats)
          (0, x0, [])
          acc
      in
      let df = (alpha *. df1) +. (0.5 *. alpha *. alpha *. df2) in
      List.rev uhats, df


  let trajectory ~theta x0 us =
    let _, xf, xs, _ = forward ~theta x0 us in
    let xs = List.rev xs |> Array.of_list |> AD.Maths.concatenate ~axis:0 in
    AD.Maths.concatenate ~axis:0 [| xs; xf |]


  let loss ~theta x0 us =
    let kf, xf, xs, us = forward ~theta x0 us in
    let fl = final_loss ~theta ~k:kf ~x:xf in
    let _, rl =
      List.fold_left2
        (fun (k, rl) x u -> pred k, AD.Maths.(rl + running_loss ~theta ~k ~x ~u))
        (kf - 1, AD.F 0.)
        xs
        us
    in
    AD.Maths.(fl + rl) |> AD.unpack_flt


  let differentiable_loss ~theta taus_f =
    let array_taus =
      taus_f
      |> fun x -> AD.Maths.split ~axis:0 (Array.init (AD.Arr.shape x).(0) (fun _ -> 1)) x
    in
    let tf = Array.length array_taus in
    let mapped =
      Array.mapi
        (fun i tau ->
          let tau = AD.Maths.reshape tau [| 1; n + m |] in
          let x, u =
            ( AD.Maths.get_slice [ []; [ 0; pred n ] ] tau
            , AD.Maths.get_slice [ []; [ n; -1 ] ] tau )
          in
          if i = pred tf
          then [| final_loss ~theta ~k:(succ i) ~x |]
          else [| running_loss ~theta ~k:(succ i) ~x ~u |])
        array_taus
    in
    AD.Maths.of_arrays mapped |> AD.Maths.sum'


  let g ~theta x0 ustars =
    let flxx, flx, tape, xf = ffb ~theta x0 ustars in
    let tau_f = AD.Maths.concatenate ~axis:1 [| xf; AD.Mat.zeros 1 m |] in
    let taus, fs, big_cs, small_cs, lambdas =
      let rec backward lambda (taus, fs, big_cs, small_cs, lambdas) = function
        | Lqr.{ x; u; a; b; rlx; rlu; rlxx; rluu; rlux } :: tl ->
          let big_ct =
            AD.Maths.concatenate
              ~axis:0
              [| AD.Maths.concatenate ~axis:1 [| rlxx; AD.Maths.(transpose rlux) |]
               ; AD.Maths.concatenate ~axis:1 [| rlux; rluu |]
              |]
          in
          let big_ct_top = AD.Maths.get_slice [ []; [ 0; pred n ] ] big_ct in
          let small_ct = AD.Maths.concatenate ~axis:1 [| rlx; rlu |] in
          let ft = AD.Maths.concatenate ~axis:0 [| a; b |] in
          let tau = AD.Maths.concatenate ~axis:1 [| x; u |] in
          let new_lambda =
            AD.Maths.((lambda *@ transpose a) + (tau *@ big_ct_top) + rlx)
          in
          backward
            new_lambda
            ( tau :: taus
            , ft :: fs
            , big_ct :: big_cs
            , small_ct :: small_cs
            , new_lambda :: lambdas )
            tl
        | [] -> taus, fs, big_cs, small_cs, lambdas
      in
      let lambda_f =
        let big_ctf_top = AD.Maths.concatenate ~axis:0 [| flxx; AD.Mat.zeros m n |] in
        AD.Maths.((tau_f *@ big_ctf_top) + flx)
      in
      let small_ctf = AD.Maths.concatenate ~axis:1 [| flx; AD.Mat.zeros 1 m |] in
      let big_ctf =
        AD.Maths.concatenate
          ~axis:0
          [| AD.Maths.concatenate ~axis:1 [| flxx; AD.Mat.zeros n m |]
           ; AD.Maths.concatenate ~axis:1 [| AD.Mat.zeros m (n + m) |]
          |]
      in
      backward
        lambda_f
        ([ tau_f ], [ AD.Mat.zeros (n + m) n ], [ big_ctf ], [ small_ctf ], [ lambda_f ])
        tape
    in
    let pack x = AD.Maths.stack ~axis:0 (Array.of_list x) in
    [| pack taus; pack fs; pack big_cs; pack small_cs; pack lambdas |]


  let learn ?(linesearch = true) ~theta ~stop x0 us =
    let rec loop iter us =
      if stop iter us
      then us
      else (
        let f0 = loss ~theta x0 us in
        let update = update ~theta x0 us in
        let f alpha =
          let us, df = update alpha in
          let fv = loss ~theta x0 us in
          fv, Some df, us
        in
        if not linesearch
        then (
          let _, _, us = f 1. in
          loop (succ iter) us)
        else (
          match Linesearch.backtrack f0 f with
          | Some us -> loop (succ iter) us
          | None    -> failwith "linesearch did not converge"))
    in
    loop 0 us


  let g2 =
    let swap_out_tape tape tau_bar =
      (* swapping out the tape *)
      let _, tape =
        List.fold_left
          (fun (k, tape) s ->
            let rlx =
              AD.Maths.reshape
                (AD.Maths.get_slice [ [ k ]; []; [ 0; pred n ] ] tau_bar)
                [| 1; n |]
            in
            let rlu =
              AD.Maths.reshape
                (AD.Maths.get_slice [ [ k ]; []; [ n; -1 ] ] tau_bar)
                [| 1; m |]
            in
            succ k, Lqr.{ s with rlx; rlu } :: tape)
          (0, [])
          (* the tape is backward in time hence we reverse it *)
          (List.rev tape)
      in
      let flx =
        AD.Maths.reshape
          (AD.Maths.get_slice [ [ -1 ]; []; [ 0; n - 1 ] ] tau_bar)
          [| 1; n |]
      in
      flx, tape
    in
    fun ~theta x0 ustars ->
      (* 
    input: [|taus; big_f; big_ct; ct; lambdas |]
    output: taus
    *)
      let ds ~tau_bar =
        (* recreating tape, pass as argument in the future *)
        let flxx, _, tape, _ = ffb ~theta x0 ustars in
        (* swap out rlx, rlu with taubars: returns tape again backwards in time *)
        let flx, tape = swap_out_tape tape tau_bar in
        (* the second LQR to get ctbars which will be backwards in time *)
        let ctbars, ctbars_xf =
          let x0 = AD.Mat.zeros 1 n in
          let acc, _ = Lqr.backward_aug flxx flx tape in
          let _, xf, taus =
            List.fold_left
              (fun (k, xi, taus) (_, _, (_K, _k), a, b) ->
                let u = AD.Maths.((xi *@ _K) + _k) in
                let tau = AD.Maths.concatenate ~axis:1 [| xi; u |] in
                let next_x = AD.Maths.((xi *@ a) + (u *@ b)) in
                succ k, next_x, tau :: taus)
              (0, x0, [])
              acc
          in
          taus, xf
        in
        let ctbar_f = AD.Maths.concatenate ~axis:1 [| ctbars_xf; AD.Mat.zeros 1 m |] in
        (* begin computing dlambdas *)
        fun ~big_fs ~big_cs ->
          let dlambda_f =
            let btrans =
              AD.Maths.get_slice [ [ -1 ]; []; [ 0; pred n ] ] big_cs
              |> fun x -> AD.Maths.reshape x [| n + m; n |]
            in
            let flx =
              AD.Maths.get_slice [ [ -1 ]; []; [ 0; pred n ] ] tau_bar
              |> fun x -> AD.Maths.reshape x [| 1; n |]
            in
            AD.Maths.((ctbar_f *@ btrans) + flx)
          in
          let _, _, dlambdas =
            List.fold_left
              (fun (k, dlambda_next, dlambdas) d ->
                let rlx =
                  AD.Maths.get_slice [ [ k ]; []; [ 0; pred n ] ] tau_bar
                  |> fun x -> AD.Maths.reshape x [| 1; n |]
                in
                let atrans =
                  AD.Maths.get_slice [ [ k ]; [ 0; pred n ]; [] ] big_fs
                  |> fun x -> AD.Maths.reshape x [| n; n |] |> AD.Maths.transpose
                in
                let btrans =
                  AD.Maths.get_slice [ [ k ]; []; [ 0; pred n ] ] big_cs
                  |> fun x -> AD.Maths.reshape x [| n + m; n |]
                in
                let new_lambda =
                  AD.Maths.((dlambda_next *@ atrans) + (d *@ btrans) + rlx)
                in
                pred k, new_lambda, dlambda_next :: dlambdas)
              (-2, dlambda_f, [ dlambda_f ])
              (* cbtars backwards in time from above *)
              ctbars
          in
          ( AD.Maths.stack ~axis:0 (Array.of_list List.(rev (ctbar_f :: ctbars)))
          , AD.Maths.stack ~axis:0 (Array.of_list dlambdas) )
      in
      let big_ft_bar ~taus ~lambdas ~dlambdas ~ctbars () =
        let tdl =
          Bmo.AD.bmm
            (AD.Maths.transpose
               ~axis:[| 0; 2; 1 |]
               (AD.Maths.get_slice [ [ 0; -2 ]; []; [] ] taus))
            (AD.Maths.get_slice [ [ 1; -1 ]; []; [] ] dlambdas)
        in
        let dtl =
          Bmo.AD.bmm
            (AD.Maths.transpose
               ~axis:[| 0; 2; 1 |]
               (AD.Maths.get_slice [ [ 0; -2 ]; []; [] ] ctbars))
            (AD.Maths.get_slice [ [ 1; -1 ]; []; [] ] lambdas)
        in
        let outpt = AD.Maths.(tdl + dtl) in
        AD.Maths.concatenate ~axis:0 [| outpt; AD.Arr.zeros [| 1; n + m; n |] |]
      in
      let big_ct_bar ~taus ~ctbars () =
        let tdt = Bmo.AD.bmm (AD.Maths.transpose ~axis:[| 0; 2; 1 |] ctbars) taus in
        AD.Maths.(F 0.5 * (tdt + transpose ~axis:[| 0; 2; 1 |] tdt))
      in
      (*check this, should be l(t+1)*)
      build_aiso
        (module struct
          let label = "g2"

          let ff a =
            let x = Array.map AD.primal' a in
            x.(0)


          let df _ _ _ _ = raise (Owl_exception.NOT_IMPLEMENTED "g2 forward mode")

          let dr idxs x _ ybar =
            let x = Array.map AD.primal x in
            let ctbars, dlambdas = ds ~tau_bar:!ybar ~big_fs:x.(1) ~big_cs:x.(2) in
            (* input bars : 
            0: taus bar; 
            1: big_f bar ; 
            2: big_ct bar; 
            3: ct bar; 
            4: lambda bars *)
            List.map
              (fun idx ->
                if idx = 0
                then !ybar
                else if idx = 1
                then big_ft_bar ~taus:x.(0) ~lambdas:x.(4) ~dlambdas ~ctbars ()
                else if idx = 2
                then big_ct_bar ~taus:x.(0) ~ctbars ()
                else if idx = 3
                then ctbars
                else dlambdas) (*not actual lambda gradient but we need this after*)
              idxs
        end : Aiso)


  let unpack a =
    let x0 = AD.Maths.get_slice [ []; [ 0; n - 1 ] ] a in
    let theta = AD.Maths.get_slice [ []; [ n; -1 ] ] a in
    x0, theta


  let g3 ~theta taus =
    (* 
    inputs: [|x0; theta|] 
    outputs: [| big_f; big_c; ct |]
    *)
    let n_steps = AD.Arr.(shape taus).(0) in
    let taus = AD.Maths.split ~axis:0 (Array.make n_steps 1) taus in
    let _, fs, big_cs, small_cs =
      Array.fold_left
        (fun (k, fs, big_cs, small_cs) tau ->
          let x =
            AD.Maths.get_slice [ []; []; [ 0; n - 1 ] ] tau
            |> fun x -> AD.Maths.reshape x [| 1; n |]
          in
          let u =
            AD.Maths.get_slice [ []; []; [ n; -1 ] ] tau
            |> fun x -> AD.Maths.reshape x [| 1; m |]
          in
          let f, big_c, small_c =
            if k = n_steps
            then (
              let f = dyn ~k ~theta ~x ~u:AD.Mat.(zeros 1 m) in
              let big_c =
                let row1 =
                  AD.Maths.concatenate
                    ~axis:1
                    [| fl_xx ~theta ~k ~x; AD.Mat.zeros n (n + m) |]
                in
                let row2 = AD.Mat.zeros (n + m) m in
                AD.Maths.concatenate ~axis:0 [| row1; row2 |]
              in
              let small_c =
                AD.Maths.concatenate ~axis:1 [| fl_x ~theta ~k ~x; AD.Mat.zeros 1 m |]
              in
              f, big_c, small_c)
            else (
              let f =
                AD.Maths.concatenate
                  ~axis:0
                  [| dyn_x ~theta ~k ~x ~u; dyn_u ~theta ~k ~x ~u |]
              and big_c =
                let rlux = rl_ux ~theta ~k ~x ~u in
                let row1 =
                  AD.Maths.concatenate
                    ~axis:1
                    [| rl_xx ~theta ~k ~x ~u; AD.Maths.transpose rlux |]
                in
                let row2 =
                  AD.Maths.concatenate ~axis:1 [| rlux; rl_uu ~theta ~k ~x ~u |]
                in
                AD.Maths.concatenate ~axis:0 [| row1; row2 |]
              and small_c =
                AD.Maths.concatenate
                  ~axis:1
                  [| rl_x ~theta ~k ~x ~u; rl_u ~theta ~k ~x ~u |]
              in
              f, big_c, small_c)
          in
          succ k, f :: fs, big_c :: big_cs, small_c :: small_cs)
        (0, [], [], [])
        taus
    in
    let pack x = AD.Maths.stack ~axis:0 (Array.of_list (List.rev x)) in
    AD.Maths.concatenate
      ~axis:2
      [| pack fs; pack big_cs; AD.Maths.(transpose ~axis:[| 0; 2; 1 |] (pack small_cs)) |]


  let g1 ~linesearch ~stop us =
    (* 
    inputs: [|x0; theta|] 
    outputs: [|taus; big_f; big_c; ct; lambdas|]
    *)
    let theta_b x y ybar =
      let _, theta = unpack x in
      let taus = AD.primal' !(y.(0)) in
      let theta = AD.primal' theta in
      let theta = AD.make_reverse theta (AD.tag ()) in
      let y' = g3 ~theta taus in
      let y'bar =
        AD.Maths.concatenate
          ~axis:2
          [| !(ybar.(1))
           ; !(ybar.(2))
           ; AD.Maths.transpose ~axis:[| 0; 2; 1 |] !(ybar.(3))
          |]
      in
      AD.reverse_prop y'bar y';
      AD.adjval theta
    in
    build_siao
      (module struct
        let label = "g1"
        let ff_f _ = failwith "not implemented"

        let ff_arr a =
          let x0, theta = unpack (AD.pack_arr a) in
          let ustars = learn ~linesearch ~theta ~stop x0 us in
          g ~theta x0 ustars


        let df _ _ _ = failwith "don't care"

        let dr x _ y ybars =
          (* ybars : taubars; big_f_bar; big_c_bar; ct_bar; dlambdas *)
          let x = AD.primal x in
          let x0bar =
            let dlambdas = !(ybars.(4)) in
            let xb = AD.Maths.(get_slice [ [ 0 ]; []; [] ] dlambdas) in
            AD.Maths.(reshape xb [| 1; n |])
          in
          let theta_bar = theta_b x y ybars in
          let taub = AD.Maths.concatenate ~axis:1 [| x0bar; theta_bar |] in
          taub
      end : Siao)


  let ilqr ?(linesearch = true) ~stop x0 theta us =
    let all = g1 ~linesearch ~stop us AD.Maths.(concatenate ~axis:1 [| x0; theta |]) in
    let ustars =
      List.init n_steps (fun i ->
          AD.Maths.get_slice [ [ i ]; []; [ n; -1 ] ] all.(0)
          |> fun y -> AD.Maths.reshape y [| 1; m |] |> AD.primal')
    in
    let x0 = AD.primal' x0 in
    let theta = AD.primal' theta in
    g2 ~theta x0 ustars all
end
