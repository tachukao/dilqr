open Owl

let () = Printexc.record_backtrace true

module AD = Algodiff.D

let g = AD.F 9.81
let dt = AD.F 1E-3

let dyn ~theta ~k:_k ~x ~u =
  let theta = theta |> AD.Maths.sum' in
  let x1 = AD.Maths.get_slice [ []; [ 0 ] ] x in
  let x2 = AD.Maths.get_slice [ []; [ 1 ] ] x in
  let b = AD.pack_arr (Mat.of_arrays [| [| 1.; 0. |] |] |> Mat.transpose) in
  let sx1 = AD.Maths.sin x1 in
  let dx2 = AD.Maths.((g * sx1) - (theta * x2) + (u *@ b)) in
  let dx = [| x2; dx2 |] |> AD.Maths.concatenate ~axis:1 in
  AD.Maths.(x + (dx * dt))


let () =
  let test_theta k u x0 theta =
    AD.jacobian (fun x -> dyn ~theta ~k ~x ~u) x0
    |> AD.Maths.transpose
    |> AD.Maths.l2norm'
  in
  let ff = (test_theta 0 (AD.Mat.zeros 1 2)) (AD.Mat.ones 1 2) in
  let module FD = Owl_algodiff_check.Make (Algodiff.D) in
  let n_samples = 10 in
  let samples, directions = FD.generate_test_samples (1, 1) n_samples in
  let threshold = 1E-4 in
  let eps = 1E-5 in
  (FD.Reverse.check ~threshold ~order:`fourth ~eps ~directions ~f:ff samples |> fst
  && FD.Forward.check ~threshold ~directions ~f:ff samples |> fst)
  |> Printf.printf "%b\n%!"
