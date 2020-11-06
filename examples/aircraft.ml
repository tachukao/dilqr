open Owl
module AD = Algodiff.D

let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d [dir]")
let in_dir = Printf.sprintf "%s/%s" dir

module P = struct
  let n = 3
  let m = 3
  let n_steps = 2000
  let dims = [ 0, 0, 0 ]
  let dt = AD.F 1E-3
  let g = AD.F 9.8
  let mu = AD.F 0.01

  let a =
    Mat.of_arrays
      [| [| -0.313; 56.7; 0. |]; [| -0.0319; -0.426; 0. |]; [| 0.; 56.7; 0. |] |]


  let b = Mat.of_arrays [| [| 0.232; 0.0203; 0. |]; [| 0.; 0.; 0. |]; [| 0.; 0.; 0. |] |]
  let __a, __b = AD.pack_arr a, AD.pack_arr b
  let c = Mat.of_arrays [| [| 0.; 0.; 1. |] |]
  let __c = AD.pack_arr c

  let dyn ~theta ~k:_k ~x ~u =
    let dx =
      AD.Maths.((__a *@ transpose x) + (__b *@ transpose u)) |> AD.Maths.transpose
    in
    AD.Maths.(x + (dx * dt))


  let dyn_x =
    let f ~theta:_theta ~k:_k ~x:_x ~u:_u =
      AD.Maths.(
        AD.Mat.of_arrays [| [| 1.; 0.; 0. |]; [| 0.; 1.; 0. |]; [| 0.; 0.; 1. |] |]
        + (__a * dt))
      (* let theta = theta |> AD.Maths.sum' in *)
      |> AD.Maths.transpose
    in
    Some f


  let dyn_u =
    let f ~theta:_theta ~k:_k ~x:_x ~u:_u =
      AD.Maths.(__b * dt)
      (* let theta = theta |> AD.Maths.sum' in *)
      |> AD.Maths.transpose
    in
    Some f


  let rl_xx = None
  let rl_ux = None
  let rl_uu = None
  let rl_u = None
  let rl_x = None
  let fl_xx = None
  let fl_x = None

  let running_loss =
    let p = AD.F 100. in
    let q = AD.F 1. in
    fun ~theta ~k:_k ~x ~u ->
      let y = AD.Maths.(__c *@ transpose x) in
      let y_ref = AD.Mat.of_arrays [| [| 0.2 |] |] |> AD.Maths.transpose in
      let dy = AD.Maths.(y - y_ref) in
      let input =
        AD.Maths.(
          dt
          * ((p * sum' (transpose dy *@ dy))
            (* + sum' (sqr theta * sum' (sqr u)) *)
            + sum' (q * sum' (sqr u))))
      in
      input


  let final_loss ~theta ~k:_k ~x =
    let y = AD.Maths.(__c *@ transpose x) in
    let y_ref = AD.Mat.of_arrays [| [| 0.2 |] |] |> AD.Maths.transpose in
    let _dy = AD.Maths.(y - y_ref) in
    AD.Maths.(AD.F 0. * sum' (sqr theta * sqr _dy))
end

module M = Dilqr.Default.Make (P)

let unpack a =
  let x0 = AD.Maths.get_slice [ []; [ 0; P.n - 1 ] ] a in
  let theta = AD.Maths.get_slice [ []; [ P.n; pred 0 ] ] a in
  x0, theta


let () =
  let stop prms =
    let _ = AD.Mat.print prms in
    let x0, theta = unpack prms in
    let _ = AD.Mat.print x0, AD.Mat.print theta in
    let cprev = ref 1E9 in
    fun k us ->
      let c = M.loss ~theta x0 us in
      let pct_change = abs_float (c -. !cprev) /. !cprev in
      if k mod 1 = 0
      then (
        Printf.printf "iter %2i | cost %.6f | pct change %.10f\n%!" k c pct_change;
        cprev := c;
        M.trajectory ~theta x0 us |> AD.unpack_arr |> Mat.save_txt ~out:(in_dir "traj1");
        us
        |> Array.of_list
        |> AD.Maths.concatenate ~axis:0
        |> AD.unpack_arr
        |> Mat.save_txt ~out:(in_dir "us"));
      pct_change < 1E-2
  in
  let f us prms =
    let x0, theta = unpack prms in
    let l prms =
      let _ = Printf.printf "before : %! " in
      let _ = AD.Mat.print (AD.adjval prms) in
      let _ = Printf.printf " %! " in
      let fin_taus = M.ilqr x0 theta ~stop:(stop prms) us in
      let _ = Printf.printf "\n after : %!" in
      let _ = AD.Mat.print (AD.adjval prms) in
      let _ = Printf.printf "%!" in
      let fin_taus = AD.primal' fin_taus in
      (* let fin_taus = AD.Arr.zeros [| P.n_steps; 1; P.n + P.m |] in *)
      let theta = AD.primal' theta in
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
    let c = l prms in
    let _ = Printf.printf "cost %f %!" (AD.unpack_flt c) in
    AD.F 0.
  in
  let max_steps = 2
  and eta = AD.F 0.0001 in
  let df us = AD.grad (f us) in
  let rec grad_descent k prms =
    if k = max_steps
    then prms
    else (
      let new_us =
        if k = 0
        then List.init P.n_steps (fun _ -> AD.Mat.zeros 1 P.m)
        else
          Mat.load_txt "results/us"
          |> fun m -> Mat.map_rows (fun x -> AD.pack_arr x) m |> Array.to_list
      in
      let dff = df new_us prms in
      let new_prms = AD.Maths.(prms - (eta * dff)) in
      let _ = Mat.save_txt ~out:"grads" (AD.unpack_arr dff) in
      let _ = Mat.save_txt ~out:"prms" (AD.unpack_arr (AD.primal' prms)) in
      grad_descent (succ k) new_prms)
  in
  grad_descent
    0
    (AD.Maths.concatenate
       ~axis:1
       [| AD.Mat.of_arrays [| [| 0.05; 0.; 2. |] |]; AD.Mat.of_arrays [| [| 1. |] |] |])
  |> ignore

(*problem in the dynamics somewhere, when theta is given the M.ilqr and the loss seem to differ? Maybe one of them
  doesn't take into account the theta value?*)
(* let test =
  let module FD = Owl_algodiff_check.Make (Algodiff.D) in
  let n_samples = 1 in
  let stop prms =
    let _ = AD.Mat.print prms in
    let x0, theta = AD.Mat.zeros 1 3, prms in
    let _ = AD.Mat.print x0, AD.Mat.print theta in
    let cprev = ref 1E9 in
    fun k us ->
      let c = M.loss ~theta x0 us in
      let pct_change = abs_float (c -. !cprev) /. !cprev in
      if k mod 1 = 0
      then (
        Printf.printf "iter %2i | cost %.6f | pct change %.10f\n%!" k c pct_change;
        cprev := c;
        M.trajectory ~theta x0 us |> AD.unpack_arr |> Mat.save_txt ~out:(in_dir "traj1");
        us
        |> Array.of_list
        |> AD.Maths.concatenate ~axis:0
        |> AD.unpack_arr
        |> Mat.save_txt ~out:(in_dir "us"));
      pct_change < 1E-3
  in
  let f us prms =
    let x0, theta = AD.Mat.zeros 1 3, prms in
    let fin_taus = M.ilqr x0 theta ~stop:(stop prms) us in
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
  let ff prms = f (List.init P.n_steps (fun _ -> AD.Mat.zeros 1 P.m)) prms in
  let samples, directions = FD.generate_test_samples (1, 1) n_samples in
  let threshold = 1. in
  let eps = 1E-5 in
  let b1, k1 =
    FD.Reverse.check ~threshold ~order:`fourth ~eps ~directions ~f:ff samples
  in
  Printf.printf "%b, %i\n%!" b1 k1 *)
