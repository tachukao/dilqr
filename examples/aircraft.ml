open Owl
module AD = Algodiff.D
open Dilqr.Misc

let () = Owl_stats_prng.init 0
let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d [dir]")
let in_dir = Printf.sprintf "%s/%s" dir

module P = struct
  type theta = AD.t

  let primal' = AD.primal'
  let n = 3
  let m = 3
  let n_steps = 1000
  let dt = AD.F 1E-3
  let g = AD.F 9.8
  let mu = AD.F 0.01

  let a =
    Mat.of_arrays
      [| [| -0.313; 3.0; 0. |]; [| -0.0319; 2.0; 0.|]; [| 1.; 0.; 2.0|]|]


  let b = Mat.of_arrays [| [|0.;  0.232; 0.0203 |]; [| 0.; 0.; 0.2 |]; [| 1.; 0.; 0.2 |]|] 
  let __a, __b = AD.pack_arr a, AD.pack_arr b
  let _ = Stdio.printf "%i %i %!" (AD.Mat.row_num __b) (AD.Mat.col_num __b)
  let c = Mat.of_arrays [| [| 1.; 1.; 1.|] |]
  let __c = AD.pack_arr c
  let alpha = AD.Mat.ones 1 3

  let dyn ~theta ~k:_ ~x ~u =
    let _cons = AD.Maths.get_slice [ []; [ 0 ] ] theta in
    (* let theta = AD.Maths.get_slice [ []; [ 0; 8 ] ] theta in
    let theta = AD.Maths.reshape theta [| 3; 3 |] in *)
    let dx = AD.Maths.((sqrt (sqr x + F 0.1) *@ __a) + (sqr (sum' theta) * u *@ __b)) in
    AD.Maths.(x + (dx * dt))


  let dyn_x = None

  (* Some
      (fun ~theta ~k:_ ~x ~u:_ ->
        let theta = AD.Maths.get_slice [ []; [ 9; -1 ] ] theta in
        let theta = AD.Maths.reshape theta [| 3; 3 |] in
        let __a = AD.Maths.(cos x *@ theta) in
        AD.Maths.((__a * dt) + AD.Mat.eye n)) *)

  let fl_x = None
  let fl_xx = None
  let l_xx = None
  let rl_x = None
  let rl_u = None
  let rl_uu = None
  let rl_ux = Some (fun ~theta:_ ~k:_ ~x:_ ~u:_ -> AD.Mat.zeros 3 3)
  let dyn_u = None
  let rl_xx = None

  (* Some
      (fun ~theta ~k:_ ~x ->
        let theta = AD.Maths.get_slice [ []; [ 0; 2 ] ] theta in
        let theta = AD.Maths.sqr theta in
        AD.Maths.(F 0. * theta * x)) *)

  let running_loss ~theta ~k:_k ~x ~u =
    (* let theta = AD.Maths.get_slice [ []; [ 0; 8 ] ] theta in
    let theta = AD.Maths.reshape theta [| 3; 3 |] in *)
    AD.Maths.(
      (* sum' (sqr (cos x)) + sum' x + (sum' (sqr theta) * (AD.F 0.1 * sum' (sqr u)))) *)
      F 10. * sum' (sqr (x)) + F 0.001 * (sum' (sqr theta) * (sum' (sqr u))))


  let final_loss ~theta ~k:_k ~x =
    (* let theta = AD.Maths.get_slice [ []; [ 0; 8 ] ] theta in
    let theta = AD.Maths.reshape theta [| 3; 3 |] in *)
    let theta = AD.Maths.sqr theta in
    ignore theta;
    AD.Maths.((sum' (sqr (x)) + sum' (sqr theta)))
end

module M = Dilqr.Default.Make (P)

let unpack a =
  let x0 = AD.Maths.get_slice [ []; [ 0; P.n - 1 ] ] a in
  let theta = AD.Maths.get_slice [ []; [ P.n; -1 ] ] a in
  x0, theta


let example () =
  let stop prms =
    let x0, theta = AD.Mat.gaussian 1 3, prms in
    let cprev = ref 1E9 in
    fun k us ->
      let c = M.loss ~theta x0 us in
      let pct_change = abs_float (c -. !cprev) /. !cprev in
      if k mod 1 = 0
      then (
        Printf.printf "iter %2i | cost %.6f | pct change %.10f\n%!" k c pct_change;
        cprev := c);
      pct_change < 1E-8
  in
  let f us prms =
    (* let x0, theta = AD.Mat.ones 1 3, prms in *)
    let x0, theta = AD.Mat.ones 1 3, prms in
    let fin_taus, _sig_xs, _sig_us  = M.ilqr ~linesearch:true ~stop:(stop prms) ~us ~x0 ~theta () in
    let _ = Stdio.printf "length sig_us : %i %!" (List.length _sig_us) in 
    let sig_us = Array.of_list (List.map (fun x -> Arr.reshape  (AD.unpack_arr x) [|1; (AD.Mat.row_num x); (AD.Mat.col_num x)|] ) _sig_us ) |> fun z -> Arr.concatenate ~axis:0 z in 
    let sig_xs = Array.of_list (List.map (fun x -> Arr.reshape  (AD.unpack_arr x) [|1; (AD.Mat.row_num x); (AD.Mat.col_num x)|] ) _sig_xs ) |> fun z -> Arr.concatenate ~axis:0 z in
    let _ =
     Arr.save_npy
      ~out:(in_tmp_dir "sig_us") sig_us;
      Arr.save_npy
      ~out:(in_tmp_dir "sig_xs") sig_xs;
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
  let n_samples = 2 in
  let stop prms =
    let x0 = AD.Mat.zeros 1 3 in
    let theta = prms in
    let cprev = ref 1E9 in
    fun k us ->
      let c = M.loss ~theta x0 us in
      let pct_change = abs_float ((c -. !cprev) /. !cprev) in
      if k mod 1 = 0
      then (
        Printf.printf "iter %2i | cost %.6f | pct change %.10f\n%!" k c pct_change;
        cprev := c);
      pct_change < 1E-8
  in
  let f us prms =
    let x0 = AD.Mat.zeros 1 3 in
    let theta = prms in
    (* let x0, theta = prms, AD.Mat.ones 1 3 in *)
    let fin_taus, _, _  = M.ilqr ~linesearch:true ~stop:(stop prms) ~us ~x0 ~theta () in
    let fin_taus =
      AD.Maths.reshape
        fin_taus
        [| (AD.Arr.shape fin_taus).(0)
         ; (AD.Arr.shape fin_taus).(1) * (AD.Arr.shape fin_taus).(2)
        |]
    in
    let _ =
      Stdio.printf "shapeee %i %i %!" (AD.Mat.row_num fin_taus) (AD.Mat.col_num fin_taus)
    in
    let _us = AD.Maths.get_slice [ []; [ 3; -1 ] ] fin_taus in
    let _us =
      List.init (AD.Mat.row_num _us) (fun i ->
          AD.primal' (AD.Maths.get_slice [ [ i ] ] _us))
    in
   M.differentiable_loss ~theta fin_taus 
    (* M.differentiable_quus ~theta x0 _us
    |> fun z -> Array.of_list z |> AD.Maths.concatenate ~axis:0 |> AD.Maths.l2norm_sqr' *)
  in
  let ff prms = f (List.init P.n_steps (fun _ -> AD.Mat.zeros 1 P.m)) prms in
  let samples, directions = FD.generate_test_samples (1, 3) n_samples in
  let threshold = 1E-5 in
  let eps = 1E-4 in
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
  example (); 
  test_grad ()
