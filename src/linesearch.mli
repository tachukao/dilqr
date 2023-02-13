val backtrack
  :  ?alpha:float
  -> ?alpha_min:float
  -> ?tau:float
  -> ?beta:float
  -> float
  -> (float -> float * float option * 'a * 'b *'c )
  ->  ('a * 'b * 'c) option
