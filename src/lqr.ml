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
  ; sig_uu : AD.t
  ; sig_xx : AD.t
  ; f : AD.t
  }

let backward flxx flx tape =
  (* let n = AD.(shape flx).(1) in  *)
  let kf = List.length tape in
  let k, vxx0, _, df1, df2, acc =
    let rec backward (delta, mu) (k, vxx, vx, df1, df2, acc) = function
      (*also save quu_inv*)
      | ({ x = _; u = _; a; b; rlx; rlu; rlxx; rluu; rlux; sig_uu = _; sig_xx = _; f = _ }
        as s)
        :: tl ->
        let at = AD.Maths.transpose a in
        let bt = AD.Maths.transpose b in
        let m = AD.Mat.row_num b in
        let qx = AD.Maths.(rlx + (vx *@ at)) in
        let qu = AD.Maths.(rlu + (vx *@ bt)) in
        (* If rlux != 0 then the matrix [rlxx, rlux^T; rlux, rluu] must be regularized,
          instead of regularizng rlxx and rluu separately as done here *)
        let rlxx_reg = Regularisation.regularize rlxx in
        let rluu_reg = Regularisation.regularize rluu in
        let qxx = AD.Maths.(rlxx_reg + (a *@ vxx *@ at)) in
        let quu = AD.Maths.(rluu_reg + (b *@ vxx *@ bt)) in
        let quu = AD.Maths.(F 0.5 * (quu + transpose quu)) in
        let qux = AD.Maths.(rlux + (b *@ vxx *@ at)) in
        let _K =
          AD.Linalg.(linsolve quu qux)
          |> AD.Maths.transpose
          |> AD.Maths.neg
        in
        let _k =
          AD.Linalg.(linsolve quu AD.Maths.(transpose qu))
          |> AD.Maths.transpose
          |> AD.Maths.neg
        in
        let vxx = AD.Maths.(qxx + transpose (_K *@ qux)) in
        let vxx = AD.Maths.((vxx + transpose vxx) / F 2.) in
        let vx = AD.Maths.(qx + (qu *@ transpose _K)) in
        let quu_inv = AD.Linalg.(linsolve quu (AD.Mat.eye m)) in
        let acc = (s, (_K, _k, vxx, quu_inv)) :: acc in
        let df1 = AD.Maths.(df1 + sum' (_k *@ quu *@ transpose _k)) in
        let df2 = AD.Maths.(df2 + sum' (_k *@ transpose quu)) in
        backward (delta, mu) (k - 1, vxx, vx, df1, df2, acc) tl
      | [] -> k, vxx, vx, df1, df2, acc
    in
    backward (1., 0.) (kf - 1, Regularisation.regularize flxx, flx, AD.F 0., AD.F 0., []) tape
  in
  assert (k = -1);
  acc, (AD.unpack_flt df1, AD.unpack_flt df2, vxx0)


let forward acc x0 p0 =
  let _, xf, _, tape =
    (*add computation of P : 
     P1 = (tranpose inv_a)*@(P_prev + C_xx) @ inv_a
     P2 = (C_uu + (transpose B)*@P1*@B))
     P - P1 - P1*@B*@P2*@(transpose B)*@P1
     Sigma_xx = inv (P + V_zz)
     Sigma_uu = KSigma_xx(transpose K) + inv (Q_uu)*)
    List.fold_left
      (fun (k, x, p_prev, tape) (s, (_K, _k, vxx, qtuu_inv)) ->
        let u = AD.Maths.((x *@ _K) + _k) in
        let n = AD.Mat.row_num s.a in
        let p_prev = AD.Maths.(F 0.5 * (p_prev + transpose p_prev)) in
        let vxx = AD.Maths.(F 0.5 * (vxx + transpose vxx)) in
        let q_xx = AD.Maths.(p_prev + vxx) in
        let q_txx = q_xx in
        let p_prev, sigma_xx =
          try p_prev, AD.Linalg.linsolve q_txx (AD.Mat.eye (AD.Mat.row_num s.a)) with
          | _ ->
            ( AD.Maths.(F 1E-3 * AD.Mat.eye n)
            , AD.Linalg.linsolve AD.Maths.(vxx + (F 1E-4 * AD.Mat.eye n)) (AD.Mat.eye n) )
        in
        let inv_a =
          AD.Linalg.linsolve AD.Maths.(s.a + (F 1E-4 * AD.Mat.eye n)) (AD.Mat.eye n)
        in
        let p1 = AD.Maths.(inv_a *@ (p_prev + s.rlxx) *@ transpose inv_a) in
        let p1 = AD.Maths.(F 0.5 * (p1 + transpose p1)) in
        let p2_inv = AD.Maths.(s.rluu + (s.b *@ p1 *@ transpose s.b)) in
        let p2_inv = AD.Maths.(F 0.5 * (p2_inv + transpose p2_inv)) in
        let p2 =
          try AD.Linalg.linsolve p2_inv (AD.Mat.eye (AD.Mat.row_num s.b)) with
          | _ ->
            AD.Linalg.linsolve
              AD.Maths.(s.rluu + (F 1E-4 * AD.Mat.eye (AD.Mat.row_num s.b)))
              (AD.Mat.eye (AD.Mat.row_num s.b))
        in
        let new_p = AD.Maths.(p1 - (p1 *@ transpose s.b *@ p2 *@ s.b *@ p1)) in
        let new_x = AD.Maths.((x *@ s.a) + (u *@ s.b)) in
        let sigma_uu = AD.Maths.((transpose _K *@ sigma_xx *@ _K) + qtuu_inv) in
        let new_s = { s with x; u; sig_uu = sigma_uu; sig_xx = sigma_xx } in
        succ k, new_x, new_p, new_s :: tape)
      (0, x0, p0, [])
      acc
  in
  xf, tape


let adjoint lambf tape =
  List.fold_left
    (fun (lamb, lambs)
         { x = _
         ; u = _
         ; a
         ; b = _
         ; rlx
         ; rlu = _
         ; rlxx = _
         ; rluu = _
         ; rlux = _
         ; sig_uu = _
         ; sig_xx = _
         ; f = _
         } ->
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
         ; sig_uu = _
         ; sig_xx = _
         } ->
      let lambs = lamb :: lambs in
      let lamb = AD.Maths.((lamb *@ transpose a) + (_x *@ _rlxx) + rlx + (_u *@ _rlux)) in
      (* let lamb = AD.Maths.((lamb *@ transpose a) + rlx) in *)
      lamb, lambs)
    (lambf, [])
    tape
