open Owl
module AD = Algodiff.D
module FD = Owl_algodiff_check.Make (AD)

let _ = Printexc.record_backtrace true
let n = 50
let n_samples = 1000

let ff x =
  let x1 =
    AD.Maths.get_slice [ []; [ 0; 1000 - 1 ] ] x
    |> fun x -> AD.Maths.reshape x [| 10; 10; 10 |]
  in
  let x2 =
    AD.Maths.get_slice [ []; [ 1000; -1 ] ] x
    |> fun x -> AD.Maths.reshape x [| 10; 10; 10 |]
  in
  let y = Bmo.AD.bmm x1 x2 in
  AD.Maths.sum' y


(* x has dimensions 1 x 2000 *)
(* let x1 = AD.Maths.reshape x [| 10; 10; 10 |] in
   let x2 = AD.Maths.transpose ~axis:[| 1; 2; 0 |] x1 in
   AD.Maths.sum' x2 *)

(* AD.Maths.get_slice [ []; [ 0; 1000 - 1 ] ] x
     |> fun x -> AD.Maths.reshape x [| 10; 10; 10 |]
   in
   let x2 =
     AD.Maths.get_slice [ []; [ 1000; -1 ] ] x
     |> fun x -> AD.Maths.reshape x [| 10; 10; 10 |]
   in
   let y = Bmo.AD.bmm x1 x2 in
   AD.Maths.sum' y *)

let samples, directions = FD.generate_test_samples (1, 2000) n_samples
let threshold = 1E-4
let eps = 1E-5

let () =
  Printf.printf "test start\n%!";
  Printf.printf
    "%.8f, %.8f\n%!"
    (ff samples.(0) |> AD.unpack_flt)
    (ff samples.(0) |> AD.unpack_flt);
  let directions = Owl.Stats.shuffle directions in
  let directions = Array.sub directions 0 10 in
  let b1, k1 =
    FD.Reverse.check ~threshold ~order:`fourth ~eps ~directions ~f:ff samples
  in
  Printf.printf "%b, %i\n%!" b1 k1;
  let b2, k2 =
    FD.Reverse.check ~threshold ~order:`fourth ~eps ~directions ~f:ff samples
  in
  Printf.printf "%b, %i\n%!" b2 k2
