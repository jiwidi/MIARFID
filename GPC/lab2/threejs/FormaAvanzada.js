/**
 *	Seminario GPC #2. Forma Basica.
 *	Dibujar formas básicas y un modelo importado
 *	Muestra el blucle tipico de inicialización, escena y render
 *
 */

// Variables de consenso
// Motor, escena y camara
var renderer, scene, camera;

// Otras globales
var esferaCubo, angulo = 0;

// Acciones
init();
loadScene();
render();

function init() {

    // Configurar el motor de render y el canvas
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(new THREE.Color(0x0000AA));
    document.getElementById("container").appendChild(renderer.domElement);

    // Escena
    scene = new THREE.Scene();

    // Camara
    var ar = window.innerWidth / window.innerHeight;
    camera = new THREE.PerspectiveCamera(40, ar, 1, 2000);
    camera.position.set(300, 300, 300);
    camera.lookAt(new THREE.Vector3(0, 60, 0));
    scene.add(camera);
}

function loadScene() {
    // Construir el grafo de escena

    // Materiales
    var material = new THREE.MeshBasicMaterial({
        color: 'red',
        wireframe: true
    });
    var material_pinzas = new THREE.MeshBasicMaterial({
        color: 'yellow',
        wireframe: true
    });

    // Geometrias
    var rotula = new THREE.SphereGeometry(20, 20, 20);
    var robot_base = new THREE.CylinderGeometry(50, 50, 15, 32);
    var eje = new THREE.CylinderGeometry(20, 20, 18, 32);
    var esparrago = new THREE.BoxGeometry(18, 120, 12);
    var disco = new THREE.CylinderGeometry(22, 22, 6, 20);
    var nervio = new THREE.BoxGeometry(4, 80, 4);
    var mano = new THREE.CylinderGeometry(15, 15, 40, 32);
    /// Construcción de las pinzas
    var mallaPinza = new THREE.Geometry();
    var coordenadas = [-18, 0, 0, 0, 0, 0, 18, 4, 0, 18, 16, 0, 0, 20, 0, -18, 20, 0, -18, 20, -4, 0, 20, -4, 18, 16, -2, 18, 4, -2, 0, 0, -4, -18, 0, -4];
    var indices = [0, 1, 4, 4, 5, 0, 1, 10, 7, 7, 4, 1, 6, 7, 10, 10, 11, 6, 6, 11, 0, 0, 5, 6, 7, 6, 5, 5, 4, 7, 0, 1, 10, 10, 11, 0,
        1, 2, 3, 3, 4, 1, 2, 9, 8, 8, 3, 2, 7, 8, 9, 9, 10, 7, 8, 7, 4, 4, 3, 8, 2, 1, 10, 10, 9, 2
    ];
    for (var i = 0; i < coordenadas.length; i += 3) {
        var v = new THREE.Vector3(coordenadas[i], coordenadas[i + 1], coordenadas[i + 2]);
        mallaPinza.vertices.push(v);
    }

    for (var i = 0; i < indices.length; i += 3) {
        var triangulo = new THREE.Face3(indices[i], indices[i + 1], indices[i + 2]);
        mallaPinza.faces.push(triangulo);
    }

    // Objetos
    var base = new THREE.Mesh(robot_base, material);
    var brazo_eje = new THREE.Mesh(eje, material);
    var brazo_esparrago = new THREE.Mesh(esparrago, material);
    var brazo_rotula = new THREE.Mesh(rotula, material);
    var antebrazo_disco = new THREE.Mesh(disco, material_pinzas);
    var antebrazo_nervio_1 = new THREE.Mesh(nervio, material);
    var antebrazo_nervio_2 = new THREE.Mesh(nervio, material);
    var antebrazo_nervio_3 = new THREE.Mesh(nervio, material);
    var antebrazo_nervio_4 = new THREE.Mesh(nervio, material);
    var antrebazo_mano = new THREE.Mesh(mano, material);
    var pinza_izquierda = new THREE.Mesh(mallaPinza, material_pinzas);
    var pinza_derecha = new THREE.Mesh(mallaPinza, material_pinzas);

    /// Objeto contenedor
    robot = new THREE.Object3D();
    brazo = new THREE.Object3D();
    antebrazo = new THREE.Object3D();

    // Transformaciones
    /// Base
    // base.position.y = 0
    // base.position.x = +
    // base.position.z = +50
    /// Brazo eje
    brazo_eje.rotation.x = Math.PI / 2;
    /// Esparrago
    brazo_esparrago.position.y = +60
    /// Rotula
    brazo_rotula.position.y = +120
    /// Disco
    antebrazo_disco.position.y = +120
    /// Nervios
    //1
    antebrazo_nervio_1.position.y = +160
    antebrazo_nervio_1.position.z = +7
    antebrazo_nervio_1.position.x = +7
    //2
    antebrazo_nervio_2.position.y = +160
    antebrazo_nervio_2.position.z = -7
    antebrazo_nervio_2.position.x = +7
    //3
    antebrazo_nervio_3.position.y = +160
    antebrazo_nervio_3.position.z = +7
    antebrazo_nervio_3.position.x = -7
    //4
    antebrazo_nervio_4.position.y = +160
    antebrazo_nervio_4.position.z = -7
    antebrazo_nervio_4.position.x = -7
    /// Mano
    antrebazo_mano.rotation.x = Math.PI / 2;
    antrebazo_mano.position.y = +200
    /// Pinzas
    var matrixRotatePinza = new THREE.Matrix4();
    var matrizTranslacionpinza_izquierda = new THREE.Matrix4();
    pinza_izquierda.matrixAutoUpdate = false;
    matrixRotatePinza.makeRotationX(-Math.PI / 2);
    matrizTranslacionpinza_izquierda.makeTranslation(25, 20, 10);
    pinza_izquierda.matrix = matrizTranslacionpinza_izquierda.multiply(matrixRotatePinza);

    var matrizRotacionpinza_derecha = new THREE.Matrix4();
    var matrizTranslacionpinza_derecha = new THREE.Matrix4();
    pinza_derecha.matrixAutoUpdate = false;
    matrizRotacionpinza_derecha.makeRotationX(Math.PI / 2);
    matrizTranslacionpinza_derecha.makeTranslation(25, -20, -10);
    pinza_derecha.matrix = matrizTranslacionpinza_derecha.multiply(matrizRotacionpinza_derecha);

    // Organizacion de la escena
    /// Base
    robot.add(base);
    base.add(brazo)
    /// Brazo
    brazo.add(brazo_eje)
    brazo.add(brazo_esparrago)
    brazo.add(brazo_rotula)
    brazo.add(antebrazo)
    /// Antebrazo
    antebrazo.add(antebrazo_disco)
    antebrazo.add(antebrazo_nervio_1)
    antebrazo.add(antebrazo_nervio_2)
    antebrazo.add(antebrazo_nervio_3)
    antebrazo.add(antebrazo_nervio_4)
    antebrazo.add(antrebazo_mano)
    /// Antebrazo mano
    antrebazo_mano.add(pinza_izquierda)
    antrebazo_mano.add(pinza_derecha)


    robot.add(new THREE.AxisHelper(1));
    scene.add(robot);
    // esferacubo.add(esfera);
    // scene.add(new THREE.AxisHelper(3));
}

function update() {
    // Variacion de la escena entre frames
}

function render() {
    // Construir el frame y mostrarlo
    requestAnimationFrame(render);
    update();
    renderer.render(scene, camera);
}