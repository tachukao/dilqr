module AD = Owl.Algodiff.D

let print_dim str x =
  let shp = AD.Arr.shape x in
  Printf.printf "%s " str;
  Array.iter (Printf.printf "%i  ") shp;
  Printf.printf "\n %!"


