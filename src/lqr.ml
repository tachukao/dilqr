open Owl
module AD = Algodiff.D

type t =
  { x : AD.t
  ; u : AD.t
  ; a : AD.t
  ; b : AD.t
  ; rlx : AD.t
  ; rlu : AD.t
  ; rlxx : AD.t
  ; rluu : AD.t
  ; rlux : AD.t
  ; f : AD.t
  }

let backward flxx flx tape =
  (* let n = AD.(shape flx).(1) in *)
  let kf = List.length tape in
  let k, _, _, df1, df2, acc, quus =
    let rec backward (delta, mu) (k, vxx, vx, df1, df2, acc, quus) = function
      | ({ x = _; u = _; a; b; rlx; rlu; rlxx; rluu; rlux; f = _ } as s) :: tl ->
        let at = AD.Maths.transpose a in
        let bt = AD.Maths.transpose b in
        let qx = AD.Maths.(rlx + (vx *@ at)) in
        let qu = AD.Maths.(rlu + (vx *@ bt)) in
        let qxx = AD.Maths.(rlxx + (a *@ vxx *@ at)) in
        let quu = AD.Maths.(rluu + (b *@ vxx *@ bt)) in
        let quu = AD.Maths.(F 0.5 * (quu + transpose quu)) in
        let qtuu = AD.Maths.(quu + (AD.F mu * (b *@ bt))) in
        let min_eval =
          qtuu |> AD.unpack_arr |> Linalg.D.eigvals |> Dense.Matrix.Z.re |> Mat.min'
        in
        if not (min_eval > 1E-8)
        then (
          if mu > 0. then Printf.printf "Regularizing... mu = %f \n%!" mu;
          backward
            (Regularisation.increase (delta, mu))
            (kf - 1, flxx, flx, AD.F 0., AD.F 0., [], [])
            tape)
        else (
          let qux = AD.Maths.(rlux + (b *@ vxx *@ at)) in
          let qtux = qux in
          let _K =
            try AD.Linalg.(linsolve qtuu qtux) |> AD.Maths.transpose |> AD.Maths.neg with
            | e ->
              Stdio.printf "error in dilqr qtux: %s %!" (Base.Exn.to_string e);
              AD.Linalg.(
                linsolve
                  AD.Maths.(qtuu + AD.Mat.eye (AD.Mat.row_num qtuu))
                  AD.Maths.(transpose qtux))
          in
          let _k =
            try
              AD.Linalg.(linsolve qtuu AD.Maths.(transpose qu))
              |> AD.Maths.transpose
              |> AD.Maths.neg
            with
            | e ->
              Stdio.printf "error in dilqr : %s %!" (Base.Exn.to_string e);
              AD.Linalg.(
                linsolve
                  AD.Maths.(qtuu + AD.Mat.eye (AD.Mat.row_num qtuu))
                  AD.Maths.(transpose qu))
          in
          let vxx = AD.Maths.(qxx + transpose (_K *@ qux)) in
          let vxx = AD.Maths.((vxx + transpose vxx) / F 2.) in
          let vx = AD.Maths.(qx + (qu *@ transpose _K)) in
          let acc = (s, (_K, _k)) :: acc in
          let df1 = AD.Maths.(df1 + sum' (_k *@ quu *@ transpose _k)) in
          let df2 = AD.Maths.(df2 + sum' (_k *@ transpose quu)) in
          backward (delta, mu) (k - 1, vxx, vx, df1, df2, acc, quu :: quus) tl)
      | [] -> k, vxx, vx, df1, df2, acc, quus
    in
    backward (1., 0.) (kf - 1, flxx, flx, AD.F 0., AD.F 0., [], []) tape
  in
  assert (k = -1);
  acc, (AD.unpack_flt df1, AD.unpack_flt df2), quus


let forward acc x0 =
  let _, xf, tape =
    List.fold_left
      (fun (k, x, tape) (s, (_K, _k)) ->
        let u = AD.Maths.((x *@ _K) + _k) in
        let new_s = { s with x; u } in
        let new_x = AD.Maths.((x *@ s.a) + (u *@ s.b)) in
        succ k, new_x, new_s :: tape)
      (0, x0, [])
      acc
  in
  xf, tape


let adjoint lambf tape =
  List.fold_left
    (fun (lamb, lambs)
         { x = _; u = _; a; b = _; rlx; rlu = _; rlxx = _; rluu = _; rlux = _; f = _ } ->
      let lambs = lamb :: lambs in
      let lamb = AD.Maths.((lamb *@ transpose a) + rlx) in
      lamb, lambs)
    (lambf, [])
    tape


let adjoint_back xf flxx flx tape =
  let lambf = AD.Maths.((xf *@ flxx) + flx) in
  List.fold_left
    (fun (lamb, lambs)
         { x = _x
         ; u = _u
         ; a
         ; b = _
         ; rlx
         ; rlu = _
         ; rlxx = _rlxx
         ; rluu = _rluu
         ; rlux = _rlux
         ; f = _
         } ->
      let lambs = lamb :: lambs in
      let lamb = AD.Maths.((lamb *@ transpose a) + (_x *@ _rlxx) + rlx + (_u *@ _rlux)) in
      (* let lamb = AD.Maths.((lamb *@ transpose a) + rlx) in *)
      lamb, lambs)
    (lambf, [])
    tape
