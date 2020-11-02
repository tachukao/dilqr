open Owl
module AD = Algodiff.D

let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d [dir]")
let in_dir = Printf.sprintf "%s/%s" dir

module P = struct
  let n = 3
  let m = 1
  let n_steps = 2000
  let dims = [ 0, 0, 0 ]
  let dt = AD.F 1E-3
  let g = AD.F 9.8
  let mu = AD.F 0.01

  let a =
    Mat.of_arrays
      [| [| -0.313; 56.7; 0. |]; [| -0.0319; -0.426; 0. |]; [| 0.; 56.7; 0. |] |]


  let b = Mat.of_arrays [| [| 0.232; 0.0203; 0. |] |] |> Mat.transpose
  let __a, __b = AD.pack_arr a, AD.pack_arr b
  let c = Mat.of_arrays [| [| 0.; 0.; 1. |] |]
  let __c = AD.pack_arr c

  let dyn ?theta:_theta ~k:_k ~x ~u =
    let dx = AD.Maths.((__a *@ transpose x) + (__b *@ u)) |> AD.Maths.transpose in
    AD.Maths.(x + (dx * dt))


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
    let p = AD.F 50. in
    let q = AD.F 1. in
    fun ?theta:_theta ~k:_k ~x ~u ->
      let y = AD.Maths.(__c *@ transpose x) in
      let y_ref = AD.Mat.of_arrays [| [| 0.; 0.; 0. |] |] |> AD.Maths.transpose in
      let dy = AD.Maths.(y - y_ref) in
      let input = AD.Maths.(p * sum' ((transpose dy *@ dy) + sum' (q * sum' (sqr u)))) in
      input


  let final_loss ?theta:_theta ~k:_k ~x:_x = AD.F 0.
end

module M = Dilqr.Default.Make (P)

let () =
  let x0 = [| [| 0.; 0.; 0.2 |] |] |> Mat.of_arrays |> AD.pack_arr in
  let us = List.init P.n_steps (fun _ -> AD.Mat.zeros 1 P.m) in
  M.trajectory x0 us |> AD.unpack_arr |> Mat.save_txt ~out:(in_dir "traj0");
  let stop theta =
    let cprev = ref 1E9 in
    fun k us ->
      let c = M.loss ~theta x0 us in
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
      let fin_taus = M.ilqr x0 theta ~stop:(stop theta) us in
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
  let max_steps = 2
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
  grad_descent
    0
    (AD.Maths.concatenate ~axis:1 [| x0; AD.Mat.of_arrays [| [| 0.02 |] |] |])
  |> ignore


let _ =
  let u = Mat.load_txt "results/us" in
  let t = Mat.load_txt "results/traj1" in
  let taus = Mat.load_txt "taus_ilqr" in
  Mat.(save_txt ~out:"test" ((t @|| u @= zeros 1 2) - taus))

(*problem in the dynamics somewhere, when theta is given the M.ilqr and the loss seem to differ? Maybe one of them
  doesn't take into account the theta value?*)
