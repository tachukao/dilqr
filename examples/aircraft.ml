open Owl
module AD = Algodiff.D

let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d [dir]")
let in_dir = Printf.sprintf "%s/%s" dir
let tmp_dir = Cmdargs.(get_string "-tmp" |> force ~usage:"-tmp [tmp dir]")
let in_tmp_dir = Printf.sprintf "%s/%s" tmp_dir

module P = struct
  let n = 3
  let m = 3
  let n_steps = 2000
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

  let dyn ~theta:_ ~k:_k ~x ~u =
    let dx =
      AD.Maths.((__a *@ transpose x) + (__b *@ transpose u)) |> AD.Maths.transpose
    in
    AD.Maths.(x + (dx * dt))


  let dyn_x = None

  (* let f ~theta:_theta ~k:_k ~x:_x ~u:_u =
      AD.Maths.(
        AD.Mat.of_arrays [| [| 1.; 0.; 0. |]; [| 0.; 1.; 0. |]; [| 0.; 0.; 1. |] |]
        + (__a * dt))
      (* let theta = theta |> AD.Maths.sum' in *)
      |> AD.Maths.transpose
    in
    Some f *)

  let dyn_u = None

  (* let f ~theta:_theta ~k:_k ~x:_x ~u:_u =
      AD.Maths.(__b * dt)
      (* let theta = theta |> AD.Maths.sum' in *)
      |> AD.Maths.transpose
    in
    Some f
 *)

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
            + sum' (sqr theta * sum' (sqr u))
            + sum' (q * sum' (sqr u))))
      in
      input


  let final_loss ~theta:_ ~k:_k ~x =
    let y = AD.Maths.(__c *@ transpose x) in
    let y_ref = AD.Mat.of_arrays [| [| 0.2 |] |] |> AD.Maths.transpose in
    let _dy = AD.Maths.(y - y_ref) in
    AD.Maths.(sum' (sqr _dy))

  (* AD.Maths.(sum' (sqr theta * sqr _dy)) *)
end

module M = Dilqr.Default.Make (P)

let unpack a =
  let x0 = AD.Maths.get_slice [ []; [ 0; P.n - 1 ] ] a in
  let theta = AD.Maths.get_slice [ []; [ P.n; pred 0 ] ] a in
  x0, theta


let example () =
  let stop prms =
    let x0, theta = unpack prms in
    let cprev = ref 1E9 in
    fun k us ->
      let c = M.loss ~theta x0 us in
      let pct_change = abs_float (c -. !cprev) /. !cprev in
      if k mod 1 = 0
      then (
        Printf.printf "iter %2i | cost %.6f | pct change %.10f\n%!" k c pct_change;
        cprev := c);
      pct_change < 1E-3
  in
  let f us prms =
    let x0, theta = unpack prms in
    let fin_taus = M.ilqr ~linesearch:true ~stop:(stop prms) x0 theta us in
    let _ =
      Mat.save_txt
        ~out:(in_tmp_dir "taus_ilqr")
        (AD.unpack_arr
           (AD.Maths.reshape
              fin_taus
              [| (AD.Arr.shape fin_taus).(0)
               ; (AD.Arr.shape fin_taus).(1) * (AD.Arr.shape fin_taus).(2)
              |]))
    in
    AD.Maths.l2norm' fin_taus
    (* M.differentiable_loss ~theta fin_taus *)
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
      Mat.save_txt ~out:(in_tmp_dir "grads") AD.(unpack_arr dff);
      let new_prms = AD.Maths.(prms - (eta * dff)) in
      grad_descent (succ k) new_prms)
  in
  grad_descent
    0
    (AD.Maths.concatenate
       ~axis:1
       [| AD.Mat.of_arrays [| [| 0.05; 0.; 2. |] |]; AD.Mat.of_arrays [| [| 1. |] |] |])
  |> ignore


(* problem in the dynamics somewhere, when theta is given the M.ilqr and the loss seem to differ? Maybe one of them doesn't take into account the theta value?*)
let test_grad () =
  let module FD = Owl_algodiff_check.Make (Algodiff.D) in
  let n_samples = 1 in
  let stop prms =
    (* let x0, theta = AD.Mat.zeros 1 3, prms in *)
    let x0, theta = prms, AD.Mat.ones 1 1 in
    let cprev = ref 1E9 in
    fun k us ->
      let c = M.loss ~theta x0 us in
      let pct_change = abs_float (c -. !cprev) /. !cprev in
      if k mod 1 = 0
      then (
        Printf.printf "iter %2i | cost %.6f | pct change %.10f\n%!" k c pct_change;
        cprev := c);
      pct_change < 1E-3
  in
  let f us prms =
    (* let x0, theta = AD.Mat.zeros 1 3, prms in *)
    let x0, theta = prms, AD.Mat.ones 1 1 in
    let fin_taus = M.ilqr ~linesearch:false ~stop:(stop prms) x0 theta us in
    AD.Maths.l2norm' fin_taus
    (* M.differentiable_loss ~theta fin_taus *)
  in
  let ff prms = f (List.init P.n_steps (fun _ -> AD.Mat.zeros 1 P.m)) prms in
  let samples, directions = FD.generate_test_samples (1, 3) n_samples in
  let threshold = 1E-4 in
  let eps = 1E-5 in
  let b1, k1 =
    FD.Reverse.check ~threshold ~order:`fourth ~eps ~directions ~f:ff samples
  in
  Printf.printf "%b, %i\n%!" b1 k1


let () =
  (* example (); *)
  test_grad ()
