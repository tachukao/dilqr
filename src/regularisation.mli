open Owl
module AD = Algodiff.D

type t = float * float

val increase : t -> t
val decrease : t -> t

val regularize : AD.t -> AD.t