module AD = Owl.Algodiff.D

let print_dim str x =
  let shp = AD.Arr.shape x in
  Printf.printf "%s " str;
  Array.iter (Printf.printf "%i  ") shp;
  Printf.printf "\n %!"


let tmp_dir = Cmdargs.(get_string "-tmp" |> force ~usage:"-tmp [tmp dir]")
let in_tmp_dir = Printf.sprintf "%s/%s" tmp_dir
