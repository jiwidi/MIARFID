(define (problem probtransporte) (:domain transporte)
(:objects CIUDAD1 - ciudad
          Ca1 - casa
          E1 E2 - estacion
          A1 - aeropuerto
          F1 F2 - furgoneta
          T1 - tren
          Av1 - avion
          P1 - paquete
          D1 - conductor
)

(:init
    (in CIUDAD1 Ca1) (in CIUDAD1 A1) (in CIUDAD1 E1) (in CIUDAD1 E2)
    (at F1 E1) (at F2 A1) (at Av1 A1) (at T1 E2) (at P1 Ca1) (at D1 E2)
    (empty F1) (empty F2)
)

(:goal (on D1 F1) )

)