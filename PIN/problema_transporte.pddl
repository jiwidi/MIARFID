;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; PROBLEMA TRANSPORTE                                      ;;;
;;; Ejercicio 1: Dominio Proposicional                       ;;;
;;;                                                          ;;;
;;; Planificacion Inteligente @ MIARFID, UPV                 ;;;
;;;                                                          ;;;
;;; Autores:                                                 ;;;
;;;         Jaime Ferrando Huertas                           ;;;
;;;         Javier Martínez Bernia                           ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem probtransporte) (:domain transporte)
(:objects CIUDAD1 CIUDAD2 CIUDAD3 - ciudad
          Ca1 Ca2 Ca3 - casa
          E1 E2 E3 E4 - estacion
          A1 A2 - aeropuerto
          F1 F2 F3 F4 - furgoneta
          T1 T2 - tren
          Av1 Av2 - avion
          P1 P2 - paquete
          D1 D2 D3 - conductor
)

(:init
    (in CIUDAD1 Ca1) (in CIUDAD1 A1) (in CIUDAD1 E1) (in CIUDAD1 E2)
    (in CIUDAD1 F1) (in CIUDAD1 F2) (in CIUDAD1 Av1) (in CIUDAD1 T1)
    (in CIUDAD1 D1) (in CIUDAD2 P1)

    (in CIUDAD2 Ca2) (in CIUDAD2 A2) (in CIUDAD2 E3) (in CIUDAD2 F3)
    (in CIUDAD2 Av2) (in CIUDAD2 D2) (in CIUDAD2 P2)

    (in CIUDAD3 Ca3) (in CIUDAD3 E4) (in CIUDAD3 F4) (in CIUDAD3 T2)
    (in CIUDAD3 D3)
)

(:goal (and
    (at P1 Ca3) (at P2 Ca3)
))

)