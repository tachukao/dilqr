open Owl
module AD = Algodiff.D

let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d [dir]")
let in_dir = Printf.sprintf "%s/%s" dir

module P = struct
  type theta = AD.t

  let primal' = AD.primal'
  let n = 2
  let m = 2
  let n_steps = 2000
  let dims = [ 0, 0 ]
  let dt = AD.F 5E-3
  let g = AD.F 9.8
  let mu = AD.F 0.01

  let dyn ~theta:_theta ~k:_k ~x ~u =
    (* let theta = theta |> AD.Maths.sum' in *)
    let x1 = AD.Maths.get_slice [ []; [ 0 ] ] x in
    let x2 = AD.Maths.get_slice [ []; [ 1 ] ] x in
    let b = AD.pack_arr (Mat.of_arrays [| [| 1.; 0. |] |] |> Mat.transpose) in
    let _sx1 = AD.Maths.sin x1 in
    (* let _theta = AD.Maths.get_slice [ []; [ 0; 3 ] ] theta in *)
    let dx2 = AD.Maths.((g * sin x1) - (sum' _theta * mu * x2) + (u *@ b)) in
    let dx = [| x2; dx2 |] |> AD.Maths.concatenate ~axis:1 in
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
    let r = Owl.Mat.(eye m *$ 1E-3) |> AD.pack_arr in
    let q = Owl.Mat.(eye m) |> AD.pack_arr in
    fun ~theta ~k:_k ~x ~u ->
      let _theta = AD.Maths.get_slice [ []; [ 0 ] ] theta in
      let input = AD.(Maths.((F 0.5 * sum' (u *@ r * u)) + (F 2. * sum' (x *@ q * x)))) in
      input


  let final_loss =
    let q = Owl.Mat.(eye n *$ 5.) |> AD.pack_arr in
    let xstar = [| [| 0.; 0. |] |] |> Mat.of_arrays |> AD.pack_arr in
    fun ~theta ~k:_k ~x ->
      let dx = AD.Maths.(xstar - x) in
      AD.Maths.(F 0. * AD.Maths.(sum' (sqr theta) * sum' (dx *@ q * dx)))
end

module M = Dilqr.Default.Make (P)

let unpack a =
  let _ = Printf.printf "calling unpack %!" in
  let x0 = AD.Maths.get_slice [ []; [ 0; P.n - 1 ] ] a in
  let theta = AD.Maths.get_slice [ []; [ P.n; pred 0 ] ] a in
  x0, theta


let test =
  let module FD = Owl_algodiff_check.Make (Algodiff.D) in
  let n_samples = 1 in
  let stop prms =
    let _ = AD.Mat.print prms in
    let x0 = AD.Mat.of_arrays [| [| 2.; 1.1 |] |] in
    let theta = prms in
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
      pct_change < 1E-6
  in
  let f us prms =
    let x0 = AD.Mat.of_arrays [| [| 2.; 1.1 |] |] in
    let theta = prms in
    let fin_taus = M.ilqr ~linesearch:false ~x0 ~theta ~stop:(stop prms) ~us () in
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
    AD.Maths.l2norm_sqr' fin_taus
  in
  let ff prms = f (List.init P.n_steps (fun _ -> AD.Mat.zeros 1 P.m)) prms in
  let samples, directions = FD.generate_test_samples (1, 3) n_samples in
  let threshold = 1E-3 in
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
