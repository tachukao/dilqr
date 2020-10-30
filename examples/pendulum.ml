open Owl
module AD = Algodiff.D

let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d [dir]")
let in_dir = Printf.sprintf "%s/%s" dir

module P = struct
  let n = 2
  let m = 2
  let n_steps = 2000
  let dims = [ 0, 0 ]
  let dt = AD.F 1E-3
  let g = AD.F 9.8
  let mu = AD.F 0.01

  let dyn ?theta:_theta ~k:_k ~x ~u =
    (* let theta =
      match _theta with
      | None        -> AD.Mat.zeros 1 1
      | Some _theta -> _theta
    in *)
    let x1 = AD.Maths.get_slice [ []; [ 0 ] ] x in
    let x2 = AD.Maths.get_slice [ []; [ 1 ] ] x in
    let b = AD.pack_arr (Mat.of_arrays [| [| 1.; 0. |] |] |> Mat.transpose) in
    let sx1 = AD.Maths.sin x1 in
    let dx2 = AD.Maths.((g * sx1) - (mu * x2) + (u *@ b)) in
    let dx = [| x2; dx2 |] |> AD.Maths.concatenate ~axis:1 in
    AD.Maths.(x + (dx * dt))


  let dyn_x =
    let f ?theta:_theta ~k:_k ~x ~u:_u =
      (* let theta =
        match _theta with
        | None        -> AD.Mat.zeros 1 1
        | Some _theta -> _theta
      in *)
      let x1 = AD.Maths.get_slice [ []; [ 0 ] ] x |> AD.Maths.sum' in
      AD.Maths.of_arrays
        [| [| AD.F 1.; dt |]
         ; [| AD.Maths.(g * cos x1 * dt); AD.Maths.(F 1. - (mu * dt)) |]
        |]
      |> AD.Maths.transpose
    in
    Some f


  let dyn_u = None
  let rl_xx = None
  let rl_ux = None
  let rl_uu = None
  let rl_u = None
  let rl_x = None
  let fl_xx = None
  let fl_x = None

  let running_loss =
    let r = Owl.Mat.(eye m *$ 1E-5) |> AD.pack_arr in
    let q = Owl.Mat.(eye m *$ 1E-10) |> AD.pack_arr in
    fun ?theta:_theta ~k:_k ~x ~u ->
      let theta =
        match _theta with
        | None   -> AD.Mat.zeros 1 1
        | Some a -> a
      in
      let input =
        AD.(
          Maths.(
            (F 0.5 * sum' (u *@ r * u))
            + (AD.F 1E-5 * sum' theta)
            + (F 0. * sum' (x *@ q * x))))
      in
      input


  let final_loss =
    let q = Owl.Mat.(eye n *$ 5.) |> AD.pack_arr in
    let xstar = [| [| 0.; 0. |] |] |> Mat.of_arrays |> AD.pack_arr in
    fun ?theta:_theta ~k:_k ~x ->
      let dx = AD.Maths.(xstar - x) in
      AD.(Maths.(F 0.5 * sum' (dx *@ q * dx)))
end

module M = Dilqr.Default.Make (P)

let () =
  let x0 = [| [| 2.; 0. |] |] |> Mat.of_arrays |> AD.pack_arr in
  let us = List.init P.n_steps (fun _ -> AD.Mat.zeros 1 P.m) in
  M.trajectory x0 us |> AD.unpack_arr |> Mat.save_txt ~out:(in_dir "traj0");
  let stop x0s theta =
    let cprev = ref 1E9 in
    fun k us ->
      let c = M.loss ~theta x0s us in
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
      pct_change < 1E-2
  in
  let unpack a =
    let x0 = AD.Maths.get_slice [ []; [ 0; P.n - 1 ] ] a in
    let theta = AD.Maths.get_slice [ []; [ P.n; pred 0 ] ] a in
    x0, theta
  in
  let f prms =
    let x0, theta = unpack prms in
    let l x0 theta =
      let fin_taus = M.ilqr x0 theta ~stop:(stop x0 theta) us in
      let _ =
        Mat.save_txt
          ~out:"taus_ilqr"
          (AD.unpack_arr
             (AD.Maths.reshape
                fin_taus
                [| (AD.Arr.shape fin_taus).(0)
                 ; (AD.Arr.shape fin_taus).(1) * (AD.Arr.shape fin_taus).(2)
                |]))
      in
      M.differentiable_loss ~theta fin_taus
    in
    let _ = Printf.printf "cost %f %!" (AD.unpack_flt (l x0 theta)) in
    l x0 theta
  in
  let max_steps = 10
  and eta = AD.F 0.0001 in
  let df = AD.grad f in
  let rec grad_descent k prms =
    if k = max_steps
    then prms
    else (
      let dff = df prms in
      let new_prms = AD.Maths.(prms - (eta * dff)) in
      let _ = Mat.save_txt ~out:"grads" (AD.unpack_arr dff) in
      let _ = Mat.save_txt ~out:"prms" (AD.unpack_arr prms) in
      grad_descent (succ k) new_prms)
  in
  grad_descent 0 (AD.Maths.concatenate ~axis:1 [| x0; AD.Mat.of_arrays [| [| 0.2 |] |] |])
  |> ignore


  (*problem in the dynamics somewhere, when theta is given the M.ilqr and the loss seem to differ? Maybe one of them
  doesn't take into account the theta value?*)