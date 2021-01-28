;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; PROBLEMA TRANSPORTE                                      ;;;
;;; Ejercicio 2: Dominio Temporal                            ;;;
;;;                                                          ;;;
;;; Planificacion Inteligente @ MIARFID, UPV                 ;;;
;;;                                                          ;;;
;;; Autores:                                                 ;;;
;;;         Jaime Ferrando Huertas                           ;;;
;;;         Javier Martínez Bernia                           ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain transporte)

(:requirements :strips :typing :equality :durative-actions :fluents)

(:types paquete loc vehiculo conductor ciudad - object
        casa estacion aeropuerto - loc
        furgoneta tren avion - vehiculo
)

(:predicates ; Para definir en que ciudad esta cada cosa
             (in ?c - ciudad
                 ?x - loc)
             ; Localizacion dentro de una ciudad
             (at ?x - (either paquete vehiculo conductor) ?l - loc)

             ; Vehiculo cargado con paquete p
             (loaded ?v - vehiculo ?p - paquete)

             ; Conductor encima de la furgoneta f
             (on ?c - conductor ?f - furgoneta)

             ; Furgoneta sin conductor
             (empty ?f - furgoneta)
)

; FUNCIONES TEMPORALES
(:functions
    (peso ?p - paquete)
    (distancia ?l1 - loc ?l2 - loc)
    (velocidad ?f - (either furgoneta conductor avion tren)) ; Se puede hacer esto?
)

; ACCIONES

; Cargar un paquete en un vehiculo
(:durative-action load
    :parameters (?v - vehiculo ?l - loc ?p - paquete)
    :duration (= ?duration (/ (peso ?p) 5)) ; Pondremos multiplos de 5 en el peso de los paquetes
    :condition (and
        (over all (at ?v ?l))
        (at start (at ?p ?l))
    )
    :effect (and
        (at start (not (at ?p ?l)))
        (at end (loaded ?v ?p))
    )
)

; Descargar un paquete de un vehiculo
(:durative-action unload
    :parameters (?v - vehiculo ?l - loc ?p - paquete)
    :duration (= ?duration (/ (peso ?p) 5))
    :condition (and
        (over all (at ?v ?l))
        (at start (loaded ?v ?p))
    )
    :effect (and
        (at end (at ?p ?l))
        (at end (not (loaded ?v ?p)))
    )
)

; Mover un avion entre dos aeropuertos
(:durative-action mover_avion
    :parameters (?a - avion ?o - aeropuerto ?d - aeropuerto ?co - ciudad ?cd - ciudad)
    :duration (= ?duration (/ (velocidad ?a) (distancia ?o ?d) ))
    :condition (and
        (at start (at ?a ?o))  ; El avion esta en el aeropuerto origen
        (over all (not (= ?co ?cd))) ; Podria ser at start simplemente (?)
        (over all (in ?co ?o)) ; Aeropuerto origen en ciudad origen - At start (?)
        (over all (in ?cd ?d)) ; Aeropuerto destino en ciudad destino - At start (?)
    )
    :effect (and
        (at start (not(at ?a ?o)))
        (at end (at ?a ?d))
    )
)

; Mover un tren entre dos estaciones
(:durative-action mover_tren
    :parameters (?t - tren ?o - estacion ?d - estacion)
    :duration (= ?duration 20)
    :condition (and
        (at start (at ?t ?o))  ; Tren en estacion origen
        (over all (not (= ?o ?d)))  ; Las estaciones son distintas - At start (?)
    )
    :effect (and
        (at start (not (at ?t ?o)))
        (at end (at ?t ?d))
    )
)

; Mover una furgoneta
(:durative-action mover_furgoneta
    :parameters (?f - furgoneta ?c - conductor ?o - loc ?d - loc ?ci - ciudad)
    :duration (= ?duration (/ (distancia ?o ?d) (velocidad ?f)))
    :condition (and
        (over all (on ?c ?f)) ; Conductor subido en furgoneta
        (at start (at ?f ?o)) ; Furgoneta en origen
        (over all (in ?ci ?o)) ; Las dos localizaciones son de la misma ciudad
        (over all (in ?ci ?d)) ; Podria ser At start simplemente?
    )
    :effect (and
        (at start (not (at ?f ?o)))
        (at end (at ?f ?d)) ; Furgoneta en posicion destino
    )
)

; Mover un conductor por una ciudad
(:durative-action mover_conductor
    :parameters (?c - conductor ?o - loc ?d - loc ?ci - ciudad)
    :duration (= ?duration (/ (distancia ?o ?d) (velocidad ?c)))
    :condition (and
        (at start (at ?c ?o)) ; Solo existira el predicado at para un conductor cuando esté a pie
        (over all (in ?ci ?o)) ; Las dos localizaciones son de la misma ciudad
        (over all (in ?ci ?d))
    )
    :effect (and
        (at start (not (at ?c ?o)))
        (at end (at ?c ?d)) ; Conductor en localizacion destino
    )
)

; Subir un conductor a una furgoneta
(:durative-action subir_conductor
    :parameters (?f - furgoneta ?c - conductor ?l - loc)
    :duration (= ?duration 2)
    :condition (and
        (over all (at ?f ?l))
        (at start (at ?c ?l))
        (at start (empty ?f)) ; Furgoneta libre (sin conductor)
    )
    :effect (and
        (at start (not (at ?c ?l))) ; Para que el conductor no pueda desplazarse a pie (si no se baja)
        (at end (on ?c ?f))
        (at start (not (empty ?f))) ; Protegemos la furgoneta para que no se suba ningun conductor
    )
)

; Bajar un conductor de una furgoneta
(:durative-action bajar_conductor
    :parameters (?f - furgoneta ?c - conductor ?l - loc)
    :duration (= ?duration 2)
    :condition (and
        (over all (at ?f ?l))
        (at start (on ?c ?f))
    )
    :effect (and
        (at end (not (on ?c ?f)))
        (at end (at ?c ?l))
        (at end (empty ?f))
    )
)

)