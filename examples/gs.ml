open Owl

module AD = Algodiff.D


(* let unpack a dims = 
  let theta, x0 = List.fold_left (fun  (ls,a) (rows,cols) -> 
  let n =rows*cols
  in let prm = 
  AD.Maths.reshape (AD.Maths.get_slice [[];[0;n -1]] a)  [|rows;cols|] 

  in (prm::ls, AD.Maths.get_slice [[];[n;-1]] a)) ([],a) dims 
  in theta, x0 
  (*is there a pop function*)

  let pack y = AD.Maths.stack ~axis:1 (Array.map (fun x -> AD.Maths.reshape  x
  [|(AD.Arr.shape x).(0);
  (AD.Arr.shape x).(1)*(AD.Arr.shape x).(2)|]) y )




 let slice y n m = 
            (*tau  : Tx1xn

      fs : Txnx(n+m)
      Cs : Tx(n+m)x(n+m)
      cs : Tx1xn*)
      let x = AD.Maths.split ~axis:1 [|n; (n+m)*n;(n+m)*(n+m);n|] y 
      in [|AD.Maths.reshape x.(0) [|1;n|];
      AD.Maths.reshape x.(1) [|n+m;n|];
      AD.Maths.reshape x.(2) [|n+m;n+m|];
      AD.Maths.reshape x.(3) [|1;n|] |]
let g3 ?theta taus = 
            (*assume it takes as input a list of taus_stars *)
            let n_steps = List.length taus
            in let pack x = AD.Maths.stack ~axis:0 (Array.of_list x) in 
            let _, fs, big_cs, small_cs = 
            (List.fold_left (fun (k, fs, bigs_cs, small_cs) tau -> 
            let f, big_c, small_c = 
            if k = n_steps then 
            (let f = AD.Mat.zeros 1 (n+m)
            
           and big_c = AD.Maths.concatenate ~axis:0
           [|AD.Maths.concatenate ~axis:1
           [|(rl_xx ?theta ~k ~x ~u); AD.Maths.transpose  (rl_ux ?theta ~k ~x ~u)|];
           AD.Maths.concatenate ~axis:1
           [|(rl_uw ?theta ~k ~x ~u); AD.Maths.transpose  (rl_uu ?theta ~k ~x ~u)|]|]
           and small_c =  AD.Maths.concatenate ~axis:1
           [|(fl_x ?theta ~k ~x );  (AD.Mat.zeros 1 m )|] in f,big_c,small_c )        
else 
            (let x , u= AD.Maths.get_slice [[0;n-1]] tau, AD.Maths.get_slice [[n;-1]] tau
            in let f = AD.Maths.concatenate ~axis:1
            [|(dyn_x ?theta ~k ~x ~u);  (dyn_u ?theta ~k ~x ~u)|]
           and big_c =[|AD.Maths.concatenate ~axis:1
           [|(rl_xx ?theta ~k ~x ~u); AD.Maths.zeros n (n+m)|];
           AD.Maths.zeros (n+m) m|]
           and small_c =  AD.Maths.concatenate ~axis:1
           [|(rl_x ?theta ~k ~x ~u);  (rl_u ?theta ~k ~x ~u)|] in f, big_c,small_c)

           in ((succ k), f::fs, big_c::big_cs, small_c::small_cs)
            ) (0, [],[],[]) taus)
            
            in [|pack fs, pack big_cs, pack small_cs|] *)
(* 
let get_x_slice taus = 
      (Array.split ~axis:2 [|n;m|] taus).(0)



      let g1 dims =         (* We need to have the dimensions of theta as a parameter there*)             ​
      build_siao (module struct
      let label = "g1"
      let ff_f _ = failwith "not implemented"
      let ff_arr a =  let x0 , theta = unpack a dims in g1 ?theta x0 
      let df _ _ _ = failwith "don't care"
       let dr x y _ ybars = 
      let x0bar = get_x_slice (ybars.(0)) in
      let theta_bar =
      let theta = AD.primal' theta in
      let taus = AD.primal' y.(0) in (* TODO: from y *)
      let theta = make_reverse theta (AD.tag())  in
      let y' = g3 ?theta taus in (* TODO g3 *)
      let y'bar = pack ybars in
       AD.Reverse.reverse_reset y'; 
       AD.Reverse.reverse_push y' y'bar;
      AD.adjval theta in
      pack x0bar thetabar 
      end : Siao) *)