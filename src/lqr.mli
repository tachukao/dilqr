open Owl
module AD = Algodiff.D

type t =
  { x : AD.t
  ; u : AD.t
  ; a : AD.t
  ; b : AD.t
  ; rlx : AD.t
  ; rlu : AD.t
  ; rlxx : AD.t
  ; rluu : AD.t
  ; rlux : AD.t
  ; f : AD.t
  }

val backward
  :  AD.t
  -> AD.t
  -> t list
  -> (t * (AD.t * AD.t)) list * (float * float) * AD.t list

val forward : (t * (AD.t * AD.t)) list -> AD.t -> AD.t * t list
val adjoint : AD.t -> t list -> AD.t * AD.t list
val adjoint_back : AD.t -> AD.t -> AD.t -> t list -> AD.t * AD.t list
