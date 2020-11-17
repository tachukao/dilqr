open Owl
module AD = Algodiff.D
open AD.Builder

type t = theta:AD.t -> k:int -> x:AD.t -> u:AD.t -> AD.t
type s = theta:AD.t -> k:int -> x:AD.t -> AD.t
type final_loss = theta:AD.t -> k:int -> x:AD.t -> AD.t
type running_loss = theta:AD.t -> k:int -> x:AD.t -> u:AD.t -> AD.t

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
    let default ~theta ~k ~x ~u = AD.jacobian (fun u -> dyn ~theta ~k ~x ~u) u in
    Option.value dyn_u ~default


  let dyn_x =
    let default ~theta ~k ~x ~u = AD.jacobian (fun x -> dyn ~theta ~k ~x ~u) x in
    Option.value dyn_x ~default


  let rl_u =
    let default ~theta ~k ~x ~u = AD.grad (fun u -> running_loss ~theta ~k ~x ~u) u in
    Option.value rl_u ~default


  let rl_x =
    let default ~theta ~k ~x ~u = AD.grad (fun x -> running_loss ~theta ~k ~x ~u) x in
    Option.value rl_x ~default


  let rl_uu =
    let default ~theta ~k ~x ~u = AD.jacobian (fun u -> rl_u ~theta ~k ~x ~u) u in
    Option.value rl_uu ~default


  let rl_xx =
    let default ~theta ~k ~x ~u = AD.jacobian (fun x -> rl_x ~theta ~k ~x ~u) x in
    Option.value rl_xx ~default


  let rl_ux =
    let default ~theta ~k ~x ~u = AD.jacobian (fun u -> rl_x ~theta ~k ~x ~u) u in
    Option.value rl_ux ~default


  let fl_x =
    let default ~theta ~k ~x = AD.grad (fun x -> final_loss ~theta ~k ~x) x in
    Option.value fl_x ~default


  let fl_xx =
    let default ~theta ~k ~x = AD.jacobian (fun x -> fl_x ~theta ~k ~x) x in
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
          (fun (k, xhat, uhats) ((s : Lqr.t), (_K, _k)) ->
            let dx = AD.Maths.(xhat - s.x) in
            let du = AD.Maths.((dx *@ _K) + (AD.F alpha * _k)) in
            let uhat = AD.Maths.(s.u + du) in
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
            succ k, Lqr.{ s with rlu; rlx } :: tape)
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
    fun ~taus ~ustars ~theta ~lambdas ->
      let ds ~x0 ~tau_bar =
        (* recreating tape, pass as argument in the future *)
        let flxx, _, tape, _ = ffb ~theta x0 ustars in
        let flx, tape = swap_out_tape tape tau_bar in
        let acc, _ = Lqr.backward flxx flx tape in
        let ctbars_xf, ctbars_tape = Lqr.forward acc AD.Mat.(zeros 1 n) in
        let dlambda0, dlambdas = Lqr.adjoint ctbars_xf flxx flx ctbars_tape in
        let ctbars =
          List.map
            (fun (s : Lqr.t) -> AD.Maths.(concatenate ~axis:1 [| s.x; s.u |]))
            ctbars_tape
          |> List.cons AD.Maths.(concatenate ~axis:1 [| ctbars_xf; AD.Mat.zeros 1 m |])
          |> List.rev
        in
        ( AD.Maths.stack ~axis:0 (Array.of_list ctbars)
        , AD.Maths.stack ~axis:0 (Array.of_list (dlambda0 :: dlambdas)) )
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
        let output = AD.Maths.(tdl + (AD.F 0.5 * dtl)) in
        AD.Maths.concatenate ~axis:0 [| output; AD.Arr.zeros [| 1; n + m; n |] |]
      in
      let big_ct_bar ~taus ~ctbars () =
        let tdt = Bmo.AD.bmm (AD.Maths.transpose ~axis:[| 0; 2; 1 |] ctbars) taus in
        (* AD.Maths.(F 0.5 * (tdt + transpose ~axis:[| 0; 2; 1 |] tdt)) *)
        tdt
      in
      build_aiso
        (module struct
          let label = "g2"
          let ff _ = AD.primal' taus
          let df _ _ _ _ = raise (Owl_exception.NOT_IMPLEMENTED "g2 forward mode")

          let dr idxs x _ ybar =
            let x0 = x.(3) in
            let ctbars, dlambdas = ds ~x0 ~tau_bar:!ybar in
            List.map
              (fun idx ->
                if idx = 0
                then AD.Maths.(F 1. * big_ft_bar ~taus ~lambdas ~dlambdas ~ctbars ())
                else if idx = 1
                then AD.Maths.(F 0.5 * big_ct_bar ~taus ~ctbars ())
                else if idx = 2
                then AD.Maths.(F 0.5 * ctbars)
                else (
                  let g = AD.Maths.(get_slice [ [ 0 ] ] dlambdas) in
                  AD.Maths.reshape g [| 1; -1 |]))
              idxs
        end : Aiso)


  let g1 ~x0 ~ustars theta =
    let flxx, flx, tape, xf = ffb ~theta x0 ustars in
    let lambda0, lambdas = Lqr.adjoint xf flxx flx tape in
    let lambdas = AD.Maths.stack ~axis:0 (Array.of_list (lambda0 :: lambdas)) in
    let big_taus = [ AD.Maths.concatenate ~axis:1 [| xf; AD.Mat.zeros 1 m |] ] in
    let big_fs = [ AD.Mat.zeros (n + m) n ] in
    let big_cs =
      let row1 = AD.Maths.(concatenate ~axis:1 [| flxx; AD.Mat.zeros n m |]) in
      let row2 = AD.Mat.zeros m (n + m) in
      [ AD.Maths.concatenate ~axis:0 [| row1; row2 |] ]
    in
    let cs = [ AD.Maths.concatenate ~axis:1 [| flx; AD.Mat.zeros 1 m |] ] in
    let big_taus, big_fs, big_cs, cs =
      List.fold_left
        (fun (taus, big_fs, big_cs, cs) (s : Lqr.t) ->
          let taus =
            let tau = AD.Maths.concatenate ~axis:1 [| s.x; s.u |] in
            tau :: taus
          in
          let big_f = AD.Maths.(concatenate ~axis:0 [| s.a; s.b |]) in
          let big_c =
            let row1 = AD.Maths.(concatenate ~axis:1 [| s.rlxx; transpose s.rlux |]) in
            let row2 = AD.Maths.(concatenate ~axis:1 [| s.rlux; s.rluu |]) in
            AD.Maths.(concatenate ~axis:0 [| row1; row2 |])
          in
          let c = AD.Maths.(concatenate ~axis:1 [| s.rlx; s.rlu |]) in
          taus, big_f :: big_fs, big_c :: big_cs, c :: cs)
        (big_taus, big_fs, big_cs, cs)
        tape
    in
    let taus = AD.Maths.stack ~axis:0 Array.(of_list big_taus) in
    let big_fs = AD.Maths.stack ~axis:0 Array.(of_list big_fs) in
    let big_cs = AD.Maths.stack ~axis:0 Array.(of_list big_cs) in
    let cs = AD.Maths.stack ~axis:0 Array.(of_list cs) in
    taus, big_fs, big_cs, cs, lambdas


  let ilqr ?(linesearch = true) ~stop ~us x0 theta =
    let ustars = learn ~linesearch ~theta:(AD.primal' theta) ~stop AD.(primal' x0) us in
    let taus, big_fs, big_cs, cs, lambdas = g1 ~x0:(AD.primal' x0) ~ustars theta in
    let _ =
      Mat.save_txt
        ~out:"tmp/taus_ilqr"
        (AD.unpack_arr (AD.Arr.reshape taus [| n_steps + 1; n + m |]));
      Mat.save_txt
        ~out:"tmp/ustars"
        (AD.unpack_arr
           (AD.Maths.concatenate
              ~axis:0
              (Array.of_list (List.map (fun x -> AD.Maths.reshape x [| 1; m |]) ustars))))
    in
    let inp = [| big_fs; big_cs; cs; x0 |] in
    g2
      ~lambdas:(AD.primal' lambdas)
      ~taus:(AD.primal' taus)
      ~ustars
      ~theta:(AD.primal' theta)
      inp
end
