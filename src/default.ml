open Owl
module AD = Algodiff.D

let _ = Printexc.record_backtrace true

type t = k:int -> theta:AD.t -> x:AD.t -> u:AD.t -> AD.t
type s = k:int -> theta:AD.t -> x:AD.t -> AD.t
type final_loss = k:int -> theta:AD.t -> x:AD.t -> AD.t
type running_loss = k:int -> theta:AD.t -> x:AD.t -> u:AD.t -> AD.t

let forward_for_backward
    ?dyn_x
    ?dyn_u
    ?rl_uu
    ?rl_xx
    ?rl_ux
    ?rl_u
    ?rl_x
    ?fl_xx
    ?fl_x
    ~dyn
    ~theta
    ~running_loss
    ~final_loss
    ()
  =
  let dyn_u =
    let default ~k ~theta ~x ~u =
      AD.jacobian (fun u -> dyn ~k ~theta ~x ~u) u |> AD.Maths.transpose
    in
    Option.value dyn_u ~default
  in
  let dyn_x =
    let default ~k ~theta ~x ~u =
      AD.jacobian (fun x -> dyn ~k ~theta ~x ~u) x |> AD.Maths.transpose
    in
    Option.value dyn_x ~default
  in
  let rl_u =
    let default ~k ~theta ~x ~u = AD.grad (fun u -> running_loss ~k ~theta ~x ~u) u in
    Option.value rl_u ~default
  in
  let rl_x =
    let default ~k ~theta ~x ~u = AD.grad (fun x -> running_loss ~k ~theta ~x ~u) x in
    Option.value rl_x ~default
  in
  let rl_uu =
    let default ~k ~theta ~x ~u =
      AD.jacobian (fun u -> rl_u ~k ~theta ~x ~u) u |> AD.Maths.transpose
    in
    Option.value rl_uu ~default
  in
  let rl_xx =
    let default ~k ~theta ~x ~u =
      AD.jacobian (fun x -> rl_x ~k ~theta ~x ~u) x |> AD.Maths.transpose
    in
    Option.value rl_xx ~default
  in
  let rl_ux =
    let default ~k ~theta ~x ~u = AD.jacobian (fun x -> rl_u ~k ~theta ~x ~u) x in
    Option.value rl_ux ~default
  in
  let fl_x =
    let default ~k ~theta ~x = AD.grad (fun x -> final_loss ~k ~theta ~x) x in
    Option.value fl_x ~default
  in
  let fl_xx =
    let default ~k ~theta ~x =
      AD.jacobian (fun x -> fl_x ~k ~theta ~x) x |> AD.Maths.transpose
    in
    Option.value fl_xx ~default
  in
  fun x0 us ->
    let kf, xf, tape =
      List.fold_left
        (fun (k, x, tape) u ->
          let a = dyn_x ~k ~theta ~x ~u
          and b = dyn_u ~k ~theta ~x ~u
          and rlx = rl_x ~k ~theta ~x ~u
          and rlu = rl_u ~k ~theta ~x ~u
          and rlxx = rl_xx ~k ~theta ~x ~u
          and rluu = rl_uu ~k ~theta ~x ~u
          and rlux = rl_ux ~k ~theta ~x ~u in
          let s = Lqr.{ x; u; a; b; rlx; rlu; rlxx; rluu; rlux } in
          let x = dyn ~k ~theta ~x ~u in
          succ k, x, s :: tape)
        (0, x0, [])
        us
    in
    let flxx = fl_xx ~theta ~x:xf ~k:kf in
    let flx = fl_x ~theta ~x:xf ~k:kf in
    flxx, flx, tape


module type P = sig
  val n : int
  val m : int
  val dyn : t
  val theta : AD.t
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

  let forward x0 us theta =
    List.fold_left
      (fun (k, x, xs, us) u ->
        let xs = x :: xs in
        let us = u :: us in
        let x = dyn ~k ~theta ~x ~u in
        succ k, x, xs, us)
      (0, x0, [], [])
      us


  let update =
    let forward_for_backward =
      forward_for_backward
        ?dyn_x
        ?dyn_u
        ?rl_uu
        ?rl_xx
        ?rl_ux
        ?rl_u
        ?rl_x
        ~dyn
        ~theta
        ~running_loss
        ~final_loss
        ()
    in
    fun x0 us ->
      (* xf, xs, us are in reverse *)
      let vxxf, vxf, tape = forward_for_backward x0 us in
      let acc, (df1, df2) = Lqr.backward vxxf vxf tape in
      fun alpha ->
        let _, _, uhats =
          List.fold_left
            (fun (k, xhat, uhats) (x, u, (_K, _k)) ->
              let dx = AD.Maths.(xhat - x) in
              let du = AD.Maths.((dx *@ _K) + (AD.F alpha * _k)) in
              let uhat = AD.Maths.(u + du) in
              let uhats = uhat :: uhats in
              let xhat = dyn ~k ~theta ~x:xhat ~u:uhat in
              succ k, xhat, uhats)
            (0, x0, [])
            acc
        in
        let df = (alpha *. df1) +. (0.5 *. alpha *. alpha *. df2) in
        List.rev uhats, df


  let trajectory x0 us =
    let _, xf, xs, _ = forward x0 us theta in
    let xs = List.rev xs |> Array.of_list |> AD.Maths.concatenate ~axis:0 in
    AD.Maths.concatenate ~axis:0 [| xs; xf |]


  let loss x0 us =
    let kf, xf, xs, us = forward x0 us theta in
    let fl = final_loss ~k:kf ~theta ~x:xf in
    let _, rl =
      List.fold_left2
        (fun (k, rl) x u -> pred k, AD.Maths.(rl + running_loss ~k ~theta ~x ~u))
        (kf - 1, AD.F 0.)
        xs
        us
    in
    AD.Maths.(fl + rl) |> AD.unpack_flt


  let g x0 u_stars =
    let dyn_u =
      let default ~k ~theta ~x ~u =
        AD.jacobian (fun u -> dyn ~k ~theta ~x ~u) u |> AD.Maths.transpose
      in
      Option.value dyn_u ~default
    in
    let dyn_x =
      let default ~k ~theta ~x ~u =
        AD.jacobian (fun x -> dyn ~k ~theta ~x ~u) x |> AD.Maths.transpose
      in
      Option.value dyn_x ~default
    in
    let rl_u =
      let default ~k ~theta ~x ~u = AD.grad (fun u -> running_loss ~k ~theta ~x ~u) u in
      Option.value rl_u ~default
    in
    let rl_x =
      let default ~k ~theta ~x ~u = AD.grad (fun x -> running_loss ~k ~theta ~x ~u) x in
      Option.value rl_x ~default
    in
    let rl_uu =
      let default ~k ~theta ~x ~u =
        AD.jacobian (fun u -> rl_u ~k ~theta ~x ~u) u |> AD.Maths.transpose
      in
      Option.value rl_uu ~default
    in
    let rl_xx =
      let default ~k ~theta ~x ~u =
        AD.jacobian (fun x -> rl_x ~k ~theta ~x ~u) x |> AD.Maths.transpose
      in
      Option.value rl_xx ~default
    in
    let rl_ux =
      let default ~k ~theta ~x ~u = AD.jacobian (fun x -> rl_u ~k ~theta ~x ~u) x in
      Option.value rl_ux ~default
    in
    let fl_x =
      let default ~k ~theta ~x = AD.grad (fun x -> final_loss ~k ~theta ~x) x in
      Option.value fl_x ~default
    in
    let fl_xx =
      let default ~k ~theta ~x =
        AD.jacobian (fun x -> fl_x ~k ~theta ~x) x |> AD.Maths.transpose
      in
      Option.value fl_xx ~default
    in
    let kf, xf, tape =
      List.fold_left
        (fun (k, x, tape) u ->
          let a = dyn_x ~k ~theta ~x ~u
          and b = dyn_u ~k ~theta ~x ~u
          and rlx = rl_x ~k ~theta ~x ~u
          and rlu = rl_u ~k ~theta ~x ~u
          and rlxx = rl_xx ~k ~theta ~x ~u
          and rluu = rl_uu ~k ~theta ~x ~u
          and rlux = rl_ux ~k ~theta ~x ~u in
          let s = Lqr.{ x; u; a; b; rlx; rlu; rlxx; rluu; rlux } in
          let x = dyn ~k ~theta ~x ~u in
          succ k, x, s :: tape)
        (0, x0, [])
        u_stars
    in
    let flxx = fl_xx ~theta ~x:xf ~k:kf in
    let flx = fl_x ~theta ~x:xf ~k:kf in
    let tau_f = AD.Maths.concatenate ~axis:1 [| xf; AD.Mat.zeros 1 m |] in
    let otaus, oa_s, ob_s, oqx, oqu, oqux, oqxx, oquu, olambdas =
      let rec backward lambda (taus, a_s, b_s, qx, qu, qux, qxx, quu, lambdas) = function
        | Lqr.{ x; u; a; b; rlx; rlu; rlxx; rluu; rlux } :: tl ->
          let big_ct =
            AD.Maths.concatenate
              ~axis:0
              [| AD.Maths.concatenate ~axis:1 [| rlxx; rlux |]
               ; AD.Maths.concatenate ~axis:1 [| rlux; rluu |]
              |]
          in
          let small_ct = AD.Maths.concatenate ~axis:1 [| rlx; rlu |] in
          let a_padded = AD.Maths.concatenate ~axis:0 [| a; AD.Mat.zeros m n |] in
          let b_padded = AD.Maths.concatenate ~axis:0 [| AD.Mat.zeros n m; b |] in
          let ft = AD.Maths.concatenate ~axis:1 [| a_padded; b_padded |] in
          let tau = AD.Maths.concatenate ~axis:1 [| x; u |] in
          let new_lambda =
            AD.Maths.(
              transpose ((ft *@ transpose lambda) + (big_ct *@ transpose tau)) + small_ct)
          in
          backward
            new_lambda
            ( tau :: taus
            , a :: a_s
            , b :: b_s
            , rlx :: qx
            , rlu :: qu
            , rlux :: qux
            , rlxx :: qxx
            , rluu :: quu
            , new_lambda :: lambdas )
            tl
        | [] -> taus, a_s, b_s, qx, qu, qux, qxx, quu, lambdas
      in
      let lambda_f =
        let big_ctf =
          AD.Maths.concatenate
            ~axis:0
            [| AD.Maths.concatenate ~axis:1 [| flxx; AD.Mat.zeros n m |]
             ; AD.Maths.concatenate ~axis:1 [| AD.Mat.zeros m n; AD.Mat.zeros m m |]
            |]
        in
        let small_ctf = AD.Maths.concatenate ~axis:1 [| flx; AD.Mat.zeros 1 m |] in
        AD.Maths.(transpose (big_ctf *@ transpose tau_f) + small_ctf)
      in
      backward
        lambda_f
        ( [ tau_f ]
        , []
        , []
        , [ flx ]
        , [ AD.Mat.zeros 1 m ]
        , [ AD.Mat.zeros m n ]
        , [ flxx ]
        , [ AD.Mat.zeros m m ]
        , [ lambda_f ] )
        tape
    in
    let pack x = AD.Maths.concatenate ~axis:0 (Array.of_list x) in
    [| pack otaus
     ; pack oa_s
     ; pack ob_s
     ; pack oqx
     ; pack oqu
     ; pack oqux
     ; pack oqxx
     ; pack oquu
     ; pack olambdas
    |]


  (*lambda is a row vector as well*)

  let learn ~stop x0 us =
    let rec loop iter us =
      if stop iter us
      then (
        let _ = g x0 us in
        us)
      else (
        let f0 = loss x0 us in
        let update = update x0 us in
        let f alpha =
          let us, df = update alpha in
          let fv = loss x0 us in
          fv, Some df, us
        in
        match Linesearch.backtrack f0 f with
        | Some us -> loop (succ iter) us
        | None    -> failwith "linesearch did not converge ")
    in
    loop 0 us
end
