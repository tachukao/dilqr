open Owl
module AD = Algodiff.D
open AD.Builder

let _ = Mat.save
let _ = Printexc.record_backtrace true

type t = theta:AD.t -> k:int -> x:AD.t -> u:AD.t -> AD.t
type s = theta:AD.t -> k:int -> x:AD.t -> AD.t
type final_loss = theta:AD.t -> k:int -> x:AD.t -> AD.t
type running_loss = theta:AD.t -> k:int -> x:AD.t -> u:AD.t -> AD.t

let print_dim str x =
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
    let _ = Printf.printf "tf %i %!" tf in
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
          let new_lambda = AD.Maths.((lambda *@ a) + (tau *@ big_ct_top) + rlx) in
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
        let big_ctf_top =
          AD.Maths.concatenate
            ~axis:0
            [| AD.Maths.concatenate ~axis:1 [| flxx; AD.Mat.zeros n m |] |]
        in
        AD.Maths.(transpose (big_ctf_top *@ transpose tau_f) + flx)
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


  let lqr_update ~theta tape tau_bar final_x =
    let get_x tau_bar =
      let f ~theta:_theta ~k ~x:_x ~u:_u =
        AD.Maths.reshape
          (AD.Maths.get_slice [ [ k ]; []; [ 0; pred n ] ] tau_bar)
          [| 1; n |]
      in
      f
    in
    let get_u tau_bar =
      let f ~theta:_theta ~k ~x:_x ~u:_u =
        AD.Maths.reshape (AD.Maths.get_slice [ [ k ]; []; [ n; -1 ] ] tau_bar) [| 1; m |]
      in
      f
    in
    let get_last_x =
      let f ~theta:_theta ~k:_ ~x:_x =
        AD.Maths.reshape
          (AD.Maths.get_slice [ [ n_steps ]; []; [ 0; n - 1 ] ] tau_bar)
          [| 1; n |]
      in
      f
    in
    let kf, _, tape =
      List.fold_left
        (fun (k, new_x, new_tape) Lqr.{ x; u; a; b; rlx; rlu; rlxx; rluu; rlux } ->
          let _ = rlx in
          let _ = rlu in
          let rlx = get_x tau_bar ~theta ~k ~x ~u
          and rlu = get_u tau_bar ~theta ~k ~x ~u in
          let x = new_x in
          let u = AD.Mat.zeros 1 m in
          let new_s = Lqr.{ x; u; a; b; rlx; rlu; rlxx; rluu; rlux } in
          let newer_x = AD.Maths.((new_x *@ a) + (u *@ b)) in
          succ k, newer_x, new_s :: new_tape)
        (0, AD.Mat.zeros 1 n, [])
        (List.rev tape)
    in
    let flxx = fl_xx ~theta ~k:kf ~x:final_x in
    let flx = get_last_x ~theta ~k:kf ~x:(AD.Mat.zeros 1 n) in
    let acc, _ = Lqr.backward_aug flxx flx tape in
    let _, xf, taus =
      List.fold_left
        (fun (k, xi, taus) (_, _, (_K, _k), a, b) ->
          let u = AD.Maths.((xi *@ _K) + _k) in
          let tau = AD.Maths.concatenate ~axis:1 [| xi; u |] in
          let next_x = AD.Maths.((xi *@ a) + (u *@ b)) in
          succ k, next_x, tau :: taus)
        (0, AD.Mat.zeros 1 n, [])
        acc
    in
    (* let flxx = fl_xx ~theta ~k:kf ~x:xf in *)
    (* let flx = get_last_x ~theta ~k:kf ~x:xf in *)
    List.rev taus, xf, flxx, flx


  (*lambda is a row vector as well*)

  let learn ~theta ~stop x0 us =
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
        match Linesearch.backtrack f0 f with
        | Some us -> loop (succ iter) us
        | None    -> failwith "linesearch did not converge")
    in
    loop 0 us


  let g2 ~theta x0 ustars =
    let _, _, tape, xfinal = ffb ~theta x0 ustars in
    let tau_bar _x _y ybar () = !ybar in
    let ds tau_bar fs cs =
      let ctbars, dxf, flxx, flx = lqr_update ~theta tape tau_bar xfinal in
      let ctbar_f = AD.Maths.concatenate ~axis:1 [| dxf; AD.Mat.zeros 1 m |] in
      let _ =
        Mat.save_txt
          ~out:"ctbar_f"
          (AD.unpack_arr (AD.Maths.reshape ctbar_f [| 1; n + m |]))
      in
      let dlambda_f =
        let big_ctf_top =
          AD.Maths.concatenate
            ~axis:0
            [| AD.Maths.concatenate ~axis:0 [| flxx; AD.Mat.zeros m n |] |]
        in
        AD.Maths.((ctbar_f *@ big_ctf_top) + flx)
      in
      let _ =
        Mat.save_txt
          ~out:"dlambda_f"
          (AD.unpack_arr (AD.Maths.reshape dlambda_f [| 1; n |]))
      in
      let _, _, dlambdas =
        List.fold_left
          (fun (k, lambda_next, lambdas) d ->
            let rlx =
              AD.Maths.get_slice [ [ k ]; []; [ 0; pred n ] ] tau_bar
              |> fun x -> AD.Maths.reshape x [| 1; n |]
            in
            let a =
              AD.Maths.get_slice [ [ k ]; [ 0; pred n ]; [] ] fs
              |> fun x -> AD.Maths.reshape x [| n; n |]
            in
            let big_ct_top =
              AD.Maths.get_slice [ [ k ]; []; [ 0; pred n ] ] cs
              |> fun x -> AD.Maths.reshape x [| n + m; n |]
            in
            let _ =
              if k = 1999
              then
                Mat.save_txt
                  ~out:"ct_top"
                  (AD.unpack_arr (AD.Maths.reshape big_ct_top [| n + m; n |]))
            in
            let new_lambda = AD.Maths.((lambda_next *@ a) + (d *@ big_ct_top) + rlx) in
            pred k, new_lambda, lambda_next :: lambdas)
          (List.length ctbars - 1, dlambda_f, [ dlambda_f ])
          (List.rev ctbars)
        (* (List.rev ctbars) *)
      in
      ( AD.Maths.stack ~axis:0 (Array.of_list (ctbars @ [ ctbar_f ]))
      , AD.Maths.stack ~axis:0 (Array.of_list dlambdas) )
    in
    let big_ct_bar x _y _ybar ctbars () =
      let tdt = Bmo.AD.bmm (AD.Maths.transpose ~axis:[| 0; 2; 1 |] ctbars) x.(0) in
      let outpt = AD.Maths.(F 0.5 * (tdt + transpose ~axis:[| 0; 2; 1 |] tdt)) in
      outpt
    in
    let big_ft_bar x _y _ybar dlambdas ctbars () =
      let _ =
        Mat.save_txt
          ~out:"tau_bars"
          (AD.unpack_arr (AD.Maths.reshape x.(0) [| 2001; n + m |]))
      in
      let tdl =
        print_dim "x0" x.(0);
        print_dim "dlambdas" dlambdas;
        Bmo.AD.bmm
          (AD.Maths.transpose
             ~axis:[| 0; 2; 1 |]
             (AD.Maths.get_slice [ [ 0; -2 ]; []; [] ] x.(0)))
          (AD.Maths.get_slice [ [ 1; -1 ]; []; [] ] dlambdas)
      in
      let dtl =
        Bmo.AD.bmm
          (AD.Maths.transpose
             ~axis:[| 0; 2; 1 |]
             (AD.Maths.get_slice [ [ 0; -2 ]; []; [] ] ctbars))
          (AD.Maths.get_slice [ [ 1; -1 ]; []; [] ] x.(4))
      in
      let outpt = AD.Maths.(tdl + dtl) in
      AD.Maths.concatenate ~axis:0 [| outpt; AD.Arr.zeros [| 1; n + m; n |] |]
    in
    (*check this, should be l(t+1)*)
    build_aiso
      (module struct
        let label = "g2"

        let ff a =
          let x = Array.map AD.primal' a in
          x.(0)


        let df _ _ _ _ = raise (Owl_exception.NOT_IMPLEMENTED "g2 forward mode")

        let dr idxs x y ybar =
          (* returns xbars as a list that correspond to idxs 
           * e.g. idxs = [0; 3; 5]
           * x0 = x.(0), x3 = x.(3)
           * then return  [x0bar; x3bar; x5bar]
           * *)
          let ctbars, dlambdas = ds !ybar x.(1) x.(2) in
          let _ =
            Mat.save_txt
              ~out:"dlambdas"
              (AD.unpack_arr
                 (AD.Maths.reshape
                    (AD.Maths.get_slice [ []; []; [] ] dlambdas)
                    [| 2001; n |]));
            Mat.save_txt
              ~out:"ctbars"
              (AD.unpack_arr
                 (AD.Maths.reshape
                    (AD.Maths.get_slice [ []; []; [] ] ctbars)
                    [| 2001; n + m |]))
          in
          List.map
            (fun idx ->
              (* let _ =
                Printf.printf "idx = %i %!" idx;
                print_dim "g2 forward idx" x.(idx)
              in *)
              let _ =
                Mat.save_txt
                  ~out:"ctbars_before"
                  (AD.unpack_arr
                     (AD.Maths.reshape
                        (AD.Maths.get_slice [ []; []; [] ] ctbars)
                        [| 2001; n + m |]))
              in
              if idx = 0
              then tau_bar x y ybar ()
              else if idx = 1
              then (
                let ftbar = big_ft_bar x y ybar dlambdas ctbars () in
                ftbar)
              else if idx = 2
              then big_ct_bar x y ybar ctbars ()
              else if idx = 3
              then ctbars
              else dlambdas) (*not actual lambda gradient but we need this after*)
            idxs
      end : Aiso)


  let unpack a =
    let x0 = AD.Maths.get_slice [ []; [ 0; n - 1 ] ] a in
    let theta = AD.Maths.get_slice [ []; [ n; pred 0 ] ] a in
    x0, theta


  let g3 ~theta taus =
    let _ = print_dim "g3 taus input" taus in
    let taus =
      AD.Maths.split ~axis:0 (Array.init (AD.Arr.shape taus).(0) (fun _ -> 1)) taus
    in
    let taus = Array.to_list taus in
    (*assume it takes as input a list of taus_stars *)
    let n_steps = List.length taus in
    let pack x = AD.Maths.stack ~axis:0 (Array.of_list x) in
    let _, fs, big_cs, small_cs =
      List.fold_left
        (fun (k, fs, big_cs, small_cs) tau ->
          let x, u =
            AD.Maths.get_slice [ []; []; [ 0; n - 1 ] ] tau
            |> fun x ->
            ( AD.Maths.reshape x [| 1; n |]
            , AD.Maths.get_slice [ []; []; [ n; -1 ] ] tau
              |> fun x -> AD.Maths.reshape x [| 1; m |] )
          in
          let f, big_c, small_c =
            if k = n_steps
            then (
              let f = AD.Mat.zeros (n + m) n
              and big_c =
                AD.Maths.concatenate
                  ~axis:0
                  [| AD.Maths.concatenate
                       ~axis:1
                       [| fl_xx ~theta ~k ~x; AD.Mat.zeros n (n + m) |]
                   ; AD.Mat.zeros (n + m) m
                  |]
              and small_c =
                AD.Maths.concatenate ~axis:1 [| fl_x ~theta ~k ~x; AD.Mat.zeros 1 m |]
              in
              let _ =
                print_dim "f " f, print_dim "big_c " big_c, print_dim "small_c " small_c
              in
              f, big_c, small_c)
            else (
              let f =
                (* let _ =
                  match theta with
                  | Some x -> print_dim "theta" x
                  | None   -> ()
                in *)
                AD.Maths.concatenate
                  ~axis:0
                  [| dyn_x ~theta ~k ~x ~u; dyn_u ~theta ~k ~x ~u |]
              and big_c =
                AD.Maths.concatenate
                  ~axis:0
                  [| AD.Maths.concatenate
                       ~axis:1
                       [| rl_xx ~theta ~k ~x ~u
                        ; AD.Maths.transpose (rl_ux ~theta ~k ~x ~u)
                       |]
                   ; AD.Maths.concatenate
                       ~axis:1
                       [| rl_ux ~theta ~k ~x ~u
                        ; AD.Maths.transpose (rl_uu ~theta ~k ~x ~u)
                       |]
                  |]
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
    AD.Maths.concatenate
      ~axis:2
      [| pack fs; pack big_cs; AD.Maths.(transpose ~axis:[| 0; 2; 1 |] (pack small_cs)) |]


  let g1 ~stop us =
    let forward_g1 ~theta ~stop x0 us =
      let ustars = learn ~theta ~stop x0 us in
      g ~theta x0 ustars
    in
    let theta_b x y ybar =
      let _, theta = unpack x in
      let theta = AD.primal' theta in
      let taus = AD.primal' !(y.(0)) in
      (*ybar(1) is ftbar*)
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
          forward_g1 ~theta ~stop x0 us


        let df _ _ _ = failwith "don't care"

        let dr x _ y ybars =
          let x0bar =
            (* let taubar = !(ybars.(0)) in *)
            let dlambdas = !(ybars.(4)) in
            let xb = AD.Maths.(get_slice [ [ 0 ]; []; [] ] dlambdas) in
            Mat.save_txt ~out:"x0bar" (AD.unpack_arr (AD.Maths.reshape xb [| 1; n |]));
            AD.Maths.reshape xb [| 1; n |]
          in
          let theta_bar =
            let tb = theta_b x y ybars in
            let _ = print_dim "tb" tb in
            Mat.save_txt ~out:"thetabar" (AD.unpack_arr tb);
            tb
          in
          AD.Maths.concatenate ~axis:1 [| x0bar; theta_bar |]
      end : Siao)


  let ilqr x0 theta ~stop us =
    let all = g1 ~stop us AD.Maths.(concatenate ~axis:1 [| x0; theta |]) in
    let ustars =
      List.init n_steps (fun i ->
          AD.Maths.get_slice [ [ i ]; []; [ n; -1 ] ] all.(0)
          |> fun y -> AD.Maths.reshape y [| 1; m |])
    in
    g2 ~theta x0 ustars all
end
