open Owl
module AD = Algodiff.D

let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d [dir]")
let in_dir = Printf.sprintf "%s/%s" dir

module P = struct
  let n = 5
  let m = 2
  let n_steps = 2000
  let dt = AD.F 1E-3
  let g = AD.F 9.8
  let mu = AD.F 0.01

  let dyn ?theta:_theta ~k:_k ~x ~u =
    let x1 = AD.Maths.get_slice [ []; [ 0 ] ] x in
    let x2 = AD.Maths.get_slice [ []; [ 1 ] ] x in
    let b = AD.pack_arr (Mat.of_arrays [| [| 1.; 0. |] |] |> Mat.transpose) in
    let sx1 = AD.Maths.sin x1 in
    let dx2 = AD.Maths.((g * sx1) - (mu * x2) + (u *@ b)) in
    let dx = [| x2; dx2; AD.Mat.zeros 1 3 |] |> AD.Maths.concatenate ~axis:1 in
    AD.Maths.(x + (dx * dt))


  let theta = AD.F 0.
  let dyn_x = None
  let dyn_u = None
  let rl_xx = None
  let rl_ux = None
  let rl_uu = None
  let rl_u = None
  let rl_x = None
  let fl_xx = None
  let fl_x = None

  let running_loss =
    let q = Owl.Mat.(eye n *$ 5.) |> AD.pack_arr in
    let xstar = [| [| 0.; 0.; 0.; 0.; 0. |] |] |> Mat.of_arrays |> AD.pack_arr in
    let r = Owl.Mat.(eye m *$ 1E-20) |> AD.pack_arr in
    fun ?theta:_theta ~k:_k ~x ~u ->
      let dx = AD.Maths.(xstar - x) in
      let input = AD.(Maths.(F 0.5 * sum' (u *@ r * u))) in
      let state = AD.(Maths.(F 0.5 * sum' (dx *@ q * dx))) in
      AD.Maths.(input + state)


  let final_loss ?theta:_theta ~k:_k ~x:_x = AD.F 0.
end

module M = Dilqr.Default.Make (P)

let () =
  let x0 = [| [| Const.pi; 0.; 0.; 0.; 0. |] |] |> Mat.of_arrays |> AD.pack_arr in
  let us = List.init P.n_steps (fun _ -> AD.Mat.zeros 1 P.m) in
  M.trajectory x0 us |> AD.unpack_arr |> Mat.save_txt ~out:(in_dir "traj0");
  let stop =
    let cprev = ref 1E9 in
    fun k us ->
      let c = M.loss x0 us in
      let pct_change = abs_float (c -. !cprev) /. !cprev in
      if k mod 1 = 0
      then (
        Printf.printf "iter %2i | cost %.6f | pct change %.10f\n%!" k c pct_change;
        cprev := c;
        M.trajectory x0 us |> AD.unpack_arr |> Mat.save_txt ~out:(in_dir "traj1");
        us
        |> Array.of_list
        |> AD.Maths.concatenate ~axis:0
        |> AD.unpack_arr
        |> Mat.save_txt ~out:(in_dir "us"));
      pct_change < 1.
  in
  M.learn ~stop x0 us |> ignore


let test =
  let x0 = [| [| Const.pi; 0.; 0.; 0.; 0. |] |] |> Mat.of_arrays |> AD.pack_arr in
  let us = List.init P.n_steps (fun _ -> AD.Mat.zeros 1 P.m) in
  M.trajectory x0 us |> AD.unpack_arr |> Mat.save_txt ~out:(in_dir "traj0");
  let stop =
    let cprev = ref 1E9 in
    fun k us ->
      let c = M.loss x0 us in
      let pct_change = abs_float (c -. !cprev) /. !cprev in
      if k mod 1 = 0
      then (
        Printf.printf "iter %2i | cost %.6f | pct change %.10f\n%!" k c pct_change;
        cprev := c;
        M.trajectory x0 us |> AD.unpack_arr |> Mat.save_txt ~out:(in_dir "traj1");
        us
        |> Array.of_list
        |> AD.Maths.concatenate ~axis:0
        |> AD.unpack_arr
        |> Mat.save_txt ~out:(in_dir "us"));
      pct_change < 1.
  in
  let output = M.g1 ~stop x0 us in
  let tau = output.(0) in
  let fs = output.(1)
  and cs = output.(2) in
  let n, m = P.n, P.m in
  let ds, dxf, flx, flxx =
    M.lqr_update (AD.Mat.zeros 1 n) (List.init P.n_steps (fun _ -> AD.Mat.zeros 1 P.m))
  in
  let dtf = AD.Maths.concatenate ~axis:1 [| dxf; AD.Mat.zeros 1 m |] in
  let dlambda_f =
    let big_ctf_top =
      AD.Maths.concatenate
        ~axis:0
        [| AD.Maths.concatenate ~axis:0 [| flxx; AD.Mat.zeros m n |] |]
    in
    AD.Maths.((dtf *@ big_ctf_top) + flx)
  in
  let _, _, dlambdas =
    List.fold_left
      (fun (k, lambda_next, lambdas) d ->
        let rlx = AD.Maths.get_slice [ [ k ]; [ 0; pred n ] ] tau in
        let a = AD.Maths.get_slice [ [ k * n; ((k + 1) * n) - 1 ]; [ 0; pred n ] ] fs in
        let big_ct_top =
          AD.Maths.get_slice
            [ [ k * (n + m); ((k + 1) * (n + m)) - 1 ]; [ 0; pred n ] ]
            cs
        in
        let new_lambda = AD.Maths.((lambda_next *@ a) + (d *@ big_ct_top) + rlx) in
        pred k, new_lambda, lambda_next :: lambdas)
      (List.length ds, dlambda_f, [])
      ds
  in
  dlambdas
