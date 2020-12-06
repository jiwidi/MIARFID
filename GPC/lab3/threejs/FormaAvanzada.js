/**
 *	Seminario GPC #3. Camara
 *	Manejar diferentes camaras, marcos y picking
 *
 */

// Variables imprescindibles
var renderer, scene, camera;

// Variables globales
var esferacubo, cubo, angulo = 0;
var l = b = -4;
var r = t = -l;
var cameraControls;
var alzado, plano_cenital, perfil;
var plano_cenital;
var L = 100;
init();
loadScene();
render();

function setCameras(ar) {
    var origen = new THREE.Vector3(0, 0, 0);
    // Camaras ortograficas
    var camaraOrtografica;
    if (ar > 1) {
        camaraOrtografica = new THREE.OrthographicCamera(-L * ar, L * ar, L, -L, 0.1, 10000);

    } else {
        camaraOrtografica = new THREE.OrthographicCamera(-L, L, L * ar, -L * ar, 0.1, 10000);

    }

    plano_cenital = camaraOrtografica.clone();
    plano_cenital.position.set(0, L * 4, 0);
    plano_cenital.lookAt(origen);
    // plano_cenital.up = new THREE.Vector3(0, 0, 0);

    // Camara perspectiva
    camera = new THREE.PerspectiveCamera(80, ar, 0.1, 2000);
    camera.position.set(200, 300, 100);
    camera.lookAt(origen);

    scene.add(plano_cenital);
    scene.add(camera);

}

function init() {

    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setClearColor(new THREE.Color(0x0000AA));
    document.getElementById('container').appendChild(renderer.domElement);
    renderer.autoClear = false;

    scene = new THREE.Scene();

    setCameras(window.innerWidth / window.innerHeight);
    // Interaccion con la camara a traves de OrbitControls.js
    cameraControls = new THREE.OrbitControls(camera, renderer.domElement);
    cameraControls.target.set(0, 0, 0);

    // Atender al evento de resize
    window.addEventListener('resize', updateAspectRatio);
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
    var robot_base = new THREE.CylinderGeometry(45, 45, 15, 32);
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

    var suelo = new THREE.PlaneGeometry(10000, 10000, 10, 10);
    var ground = new THREE.Mesh(suelo, material);

    ground.rotation.x = -Math.PI / 2;

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
    // base.position.x = +50
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
    ///Suelo
    scene.add(ground);
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

function updateAspectRatio() {
    //  Renovar las dimensiones del viewPort y la matriz de proyeccion
    renderer.setSize(window.innerWidth, window.innerHeight);

    aspectRatio = window.innerWidth / window.innerHeight;
    camera.aspect = aspectRatio;
    camera.updateProjectionMatrix();

    plano_cenital.aspect = 1
    plano_cenital.updateProjectionMatrix();
}

function update() {
    cameraControls.update();
}

function render() {
    requestAnimationFrame(render);
    update();
    // renderer.clear();
    renderer.setViewport(0, 0, window.innerWidth, window.innerHeight);
    renderer.render(scene, camera);
    // Plato cenital
    renderer.setViewport(0, 0, window.innerHeight / 4, window.innerHeight / 4);
    renderer.render(scene, plano_cenital);
}