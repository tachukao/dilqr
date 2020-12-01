open Owl
module AD = Algodiff.D
open Dilqr.Misc

let () = Owl_stats_prng.init 0
let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d [dir]")
let in_dir = Printf.sprintf "%s/%s" dir

module P = struct
  let n = 3
  let m = 3
  let n_steps = 200
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
  let alpha = AD.Mat.ones 1 3

  let dyn ~theta ~k:_ ~x ~u =
    let cons = AD.Maths.get_slice [ []; [ 3; 3 + n - 1 ] ] theta in
    let theta = AD.Maths.get_slice [ []; [ 9; -1 ] ] theta in
    let theta = AD.Maths.reshape theta [| 3; 3 |] in
    let dx = AD.Maths.((x *@ __a) + (u *@ theta)) in
    AD.Maths.(x + (dx * dt) + cons)


  let dyn_x = None

  (* Some
      (fun ~theta ~k:_ ~x:_ ~u:_ ->
        ignore theta;
        let theta = AD.Maths.sqr theta in
        let __a = AD.Maths.(__a * theta) in
        AD.Maths.((__a * dt) + AD.Mat.eye n)) *)

  let fl_x = None
  let fl_xx = None
  let l_xx = None
  let rl_x = None
  let rl_u = None
  let rl_uu = None
  let rl_ux = None
  let dyn_u = None
  let rl_xx = None

  (* Some
      (fun ~theta ~k:_ ~x ->
        let theta = AD.Maths.get_slice [ []; [ 0; 2 ] ] theta in
        let theta = AD.Maths.sqr theta in
        AD.Maths.(F 0. * theta * x)) *)

  let running_loss ~theta ~k:_k ~x ~u =
    let theta = AD.Maths.get_slice [ []; [ 0; 8 ] ] theta in
    let theta = AD.Maths.reshape theta [| 3; 3 |] in
    AD.Maths.(
      sum' (sqr (x *@ theta))
      + (sum' (sqr theta) * (AD.F 0.1 * sum' (sqr (u *@ theta))))
      + sum' (sqr (x *@ theta * u)))


  let final_loss ~theta ~k:_k ~x =
    let theta = AD.Maths.get_slice [ []; [ 0; 8 ] ] theta in
    let theta = AD.Maths.reshape theta [| 3; 3 |] in
    let theta = AD.Maths.sqr theta in
    ignore theta;
    AD.Maths.(F 1. * (sum' (sqr (x *@ theta)) + sum' (sqr theta)))
end

module M = Dilqr.Default.Make (P)

let unpack a =
  let x0 = AD.Maths.get_slice [ []; [ 0; P.n - 1 ] ] a in
  let theta = AD.Maths.get_slice [ []; [ P.n; -1 ] ] a in
  x0, theta


let example () =
  let stop prms =
    let x0, theta = AD.Mat.ones 1 3, prms in
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
    (* let x0, theta = AD.Mat.ones 1 3, prms in *)
    let x0, theta = AD.Mat.ones 1 3, prms in
    let fin_taus = M.ilqr ~linesearch:false ~stop:(stop prms) ~us ~x0 ~theta () in
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
  let max_steps = 1
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
  grad_descent 0 AD.Maths.(F 0.1 * AD.Mat.gaussian 1 18) |> ignore


(* problem in the dynamics somewhere, when theta is given the M.ilqr and the loss seem to differ? Maybe one of them doesn't take into account the theta value?*)
let test_grad () =
  let module FD = Owl_algodiff_check.Make (Algodiff.D) in
  let n_samples = 1 in
  let stop prms =
    let x0, _ = unpack prms in
    let theta = prms in
    let cprev = ref 1E9 in
    fun k us ->
      let c = M.loss ~theta x0 us in
      let pct_change = abs_float (c -. !cprev) /. !cprev in
      if k mod 1 = 0
      then (
        Printf.printf "iter %2i | cost %.6f | pct change %.10f\n%!" k c pct_change;
        cprev := c);
      pct_change < 1E-12
  in
  let f us prms =
    let x0, _ = unpack prms in
    let theta = prms in
    (* let x0, theta = prms, AD.Mat.ones 1 3 in *)
    let fin_taus = M.ilqr ~linesearch:false ~stop:(stop prms) ~us ~x0 ~theta () in
    AD.Maths.sum fin_taus
  in
  let ff prms = f (List.init P.n_steps (fun _ -> AD.Mat.zeros 1 P.m)) prms in
  let samples, directions = FD.generate_test_samples (1, 18) n_samples in
  let threshold = 1E-5 in
  let eps = 1E-5 in
  let b1, k1 =
    FD.Reverse.check
      ~verbose:true
      ~threshold
      ~order:`second
      ~eps
      ~directions
      ~f:ff
      samples
  in
  Printf.printf "%b, %i\n%!" b1 k1


let () =
  (* example (); *)
  test_grad ()
