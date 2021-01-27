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

    (= (peso P1) 20)
    (= (distancia Ca1 E1) 50) (= (distancia E1 Ca1) 50)
    (= (distancia Ca1 E2) 120) (= (distancia E2 Ca1) 120)
    (= (distancia Ca1 A1) 50) (= (distancia A1 Ca1) 50)
    (= (distancia A1 E1) 100) (= (distancia E1 A1) 100)
    (= (distancia A1 E2) 60) (= (distancia E2 A1) 60)
    (= (distancia E2 E1) 50) (= (distancia E1 E2) 50)
    (= (velocidad F1) 50) (= (velocidad F2) 50) 
    (= (velocidad D1) 10)
)

(:goal (and 
    (at P1 E1) (at F2 Ca1)
))

(:metric minimize (total-time))

)