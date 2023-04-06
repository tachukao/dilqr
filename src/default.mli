open Owl
module AD = Algodiff.D
open Bmo

type 'a t = theta:'a -> k:int -> x:AD.t -> u:AD.t -> AD.t
type 'a s = theta:'a -> k:int -> x:AD.t -> AD.t
type 'a final_loss = theta:'a -> k:int -> x:AD.t -> AD.t
type 'a running_loss = theta:'a -> k:int -> x:AD.t -> u:AD.t -> AD.t

val forward_for_backward
  :  theta:'a
  -> dyn_x:'a t
  -> dyn_u:'a t
  -> rl_uu:'a t
  -> rl_xx:'a t
  -> rl_ux:'a t
  -> rl_u:'a t
  -> rl_x:'a t
  -> fl_xx:'a s
  -> fl_x:'a s
  -> dyn:'a t
  -> unit
  -> AD.t
  -> AD.t list
  -> AD.t * AD.t * Lqr.t list * AD.t * AD.t

module type P = sig
  type theta

  val primal' : theta -> theta
  val n : int
  val m : int
  val n_steps : int
  val dyn : theta t
  val final_loss : theta final_loss
  val running_loss : theta running_loss
  val dyn_x : theta t option
  val dyn_u : theta t option
  val rl_uu : theta t option
  val rl_xx : theta t option
  val rl_ux : theta t option
  val rl_u : theta t option
  val rl_x : theta t option
  val fl_xx : theta s option
  val fl_x : theta s option
end

module Make (P : P) : sig
  val trajectory : theta:P.theta -> AD.t -> AD.t list -> AD.t
  val loss : theta:P.theta -> AD.t -> AD.t list -> float
  val differentiable_loss : theta:P.theta -> AD.t -> AD.t

  val learn
    :  ?linesearch:bool
    -> theta:P.theta
    -> stop:(int -> AD.t list -> bool)
    -> AD.t
    -> AD.t list
    -> AD.t list * AD.t list * AD.t list

  val ilqr
    :  ?linesearch:bool
    -> theta:P.theta
    -> stop:(int -> AD.t list -> bool)
    -> us:AD.t list
    -> x0:AD.t
    -> unit
    -> AD.t * AD.t list * AD.t list

  val g1
    :  theta:P.theta
    -> x0:AD.t
    -> ustars:AD.t list
    -> AD.t * AD.t * AD.t * AD.t * AD.t * AD.t
end
