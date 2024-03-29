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
          P1 P2 P3 P4 P5 - paquete
          D1 D2 D3 - conductor
)

(:init

    (in CIUDAD1 Ca1) (in CIUDAD1 A1) (in CIUDAD1 E1) (in CIUDAD1 E2)
    (at F1 Ca1) (at F2 Ca1) (at Av1 A1) (at T1 E2) (at P1 Ca1) (at D1 E2)
    (at P3 E1)
    (empty F1) (empty F2)

    (in CIUDAD2 Ca2) (in CIUDAD2 A2) (in CIUDAD2 E3)
    (at F3 E3) (at Av2 A2) (at P2 Ca2) (at D2 Ca2) (at P4 E3)
    (empty F3)

    (in CIUDAD3 Ca3) (in CIUDAD3 E4)
    (at F4 Ca3) (at T2 E4) (at D3 Ca3) (at P5 Ca3)
    (empty F4)
)

(:goal (and
    (at P1 Ca2) (at P2 Ca3) (at P3 E3) (at P4 Ca3) (at P5 Ca1)
))

)