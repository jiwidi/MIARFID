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
setupGUI();
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

    robotController = {
        giro1: 0,
        giro2: 0,
        giro3: 0,
        giro4: 0,
        giro5: 0,
        separacion: 0
    }

    window.addEventListener('keydown', function movekey(event) {

        if (event.keyCode == 39 || event.keyCode == 68) {
            robot.position.x += 10;
        } else if ((event.keyCode == 38 || event.keyCode == 87)) {
            robot.position.z -= 10;
        } else if ((event.keyCode == 37 || event.keyCode == 65)) {
            robot.position.x -= 10;
        } else if ((event.keyCode == 40 || event.keyCode == 83)) {
            robot.position.z += 10;
        }
    }, false);
}



function loadScene() {
    // Construir el grafo de escena

    // Materiales
    var material = new THREE.MeshBasicMaterial({
        color: 'red',
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
    var suelo = new THREE.PlaneGeometry(1000, 1000, 50, 50)
    var ground = new THREE.Mesh(suelo, material);

    ground.rotation.x = -Math.PI / 2;

    // Objetos
    base = new THREE.Mesh(robot_base, material);
    brazo_eje = new THREE.Mesh(eje, material);
    brazo_esparrago = new THREE.Mesh(esparrago, material);
    brazo_rotula = new THREE.Mesh(rotula, material);
    antebrazo_disco = new THREE.Mesh(disco, material);
    antebrazo_nervio_1 = new THREE.Mesh(nervio, material);
    antebrazo_nervio_2 = new THREE.Mesh(nervio, material);
    antebrazo_nervio_3 = new THREE.Mesh(nervio, material);
    antebrazo_nervio_4 = new THREE.Mesh(nervio, material);
    antebrazo_mano = new THREE.Mesh(mano, material);

    //Geometria de la pinza
    geopinza = new THREE.Geometry();
    geopinza.vertices.push(
        new THREE.Vector3(0, -8, -10), //0
        new THREE.Vector3(19, -8, -10), //1
        new THREE.Vector3(0, -8, 10), //2
        new THREE.Vector3(19, -8, 10), //3
        new THREE.Vector3(0, -12, -10), //4
        new THREE.Vector3(19, -12, -10), //5
        new THREE.Vector3(0, -12, 10), //6
        new THREE.Vector3(19, -12, 10), //7
        new THREE.Vector3(38, -8, -5), //8
        new THREE.Vector3(38, -12, -5), //9
        new THREE.Vector3(38, -8, 5), //10
        new THREE.Vector3(38, -12, 5), //11
    );
    geopinza.faces.push(
        new THREE.Face3(0, 3, 2),
        new THREE.Face3(0, 1, 3),
        new THREE.Face3(1, 7, 3),
        new THREE.Face3(1, 5, 7),
        new THREE.Face3(5, 6, 7),
        new THREE.Face3(5, 4, 6),
        new THREE.Face3(4, 2, 6),
        new THREE.Face3(4, 0, 2),
        new THREE.Face3(2, 7, 6),
        new THREE.Face3(2, 3, 7),
        new THREE.Face3(4, 1, 0),
        new THREE.Face3(4, 5, 1),
        new THREE.Face3(1, 10, 3),
        new THREE.Face3(1, 8, 10),
        new THREE.Face3(8, 11, 10),
        new THREE.Face3(8, 9, 11),
        new THREE.Face3(9, 7, 11),
        new THREE.Face3(9, 5, 7),
        new THREE.Face3(3, 11, 7),
        new THREE.Face3(3, 10, 11),
        new THREE.Face3(5, 8, 1),
        new THREE.Face3(5, 9, 8),
    )
    pinza_izquierda = new THREE.Mesh(geopinza, material);
    pinza_derecha = new THREE.Mesh(geopinza, material);



    /// Objeto contenedor
    robot = new THREE.Object3D();
    brazo = new THREE.Object3D();
    antebrazo = new THREE.Object3D();

    // Transformaciones
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
    antebrazo_nervio_1.position.y = +40
    antebrazo_nervio_1.position.z = +7
    antebrazo_nervio_1.position.x = +7
    //2
    antebrazo_nervio_2.position.y = +40
    antebrazo_nervio_2.position.z = -7
    antebrazo_nervio_2.position.x = +7
    //3
    antebrazo_nervio_3.position.y = +40
    antebrazo_nervio_3.position.z = +7
    antebrazo_nervio_3.position.x = -7
    //4
    antebrazo_nervio_4.position.y = +40
    antebrazo_nervio_4.position.z = -7
    antebrazo_nervio_4.position.x = -7
    /// Mano
    antebrazo_mano.rotation.x = Math.PI / 2;
    antebrazo_mano.position.z = +7
    antebrazo_mano.position.x = +7
    antebrazo_mano.position.y = +40
    // /// Pinzas
    // pinza_derecha.rotateY(Math.PI / 2)
    pinza_izquierda.position.set(0, -10, 0)
    pinza_derecha.position.set(0, 30, 0)


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
    antebrazo_disco.add(antebrazo_nervio_1)
    antebrazo_disco.add(antebrazo_nervio_2)
    antebrazo_disco.add(antebrazo_nervio_3)
    antebrazo_disco.add(antebrazo_nervio_4)
    antebrazo.add(antebrazo_disco)
    // antebrazo.add(antebrazo_nervio_1)
    // antebrazo.add(antebrazo_nervio_2)
    // antebrazo.add(antebrazo_nervio_3)
    // antebrazo.add(antebrazo_nervio_4)
    antebrazo_nervio_4.add(antebrazo_mano)
    /// Antebrazo mano
    antebrazo_mano.add(pinza_izquierda)
    antebrazo_mano.add(pinza_derecha)


    robot.add(new THREE.AxisHelper(1));
    scene.add(robot);
    // scene.add(new THREE.AxesHelper(1000));
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
    base.rotation.y = effectControl.giroBase * Math.PI / 180
    brazo.rotation.z = effectControl.giroBrazo * Math.PI / 180
    antebrazo.rotation.y = effectControl.giroAntebrazoZ * Math.PI / 180
    antebrazo_disco.rotation.z = effectControl.giroAntebrazoY * Math.PI / 180
    antebrazo_mano.rotation.y = effectControl.giroPinzaZ * Math.PI / 180
    pinza_izquierda.position.y = 3 - effectControl.aperturaPinza
    pinza_derecha.position.y = 18 + effectControl.aperturaPinza
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

function setupGUI() {

    //Interfaz de usuario
    effectControl = {
        giroBase: 0,
        giroBrazo: 0,
        giroAntebrazoY: 0,
        giroAntebrazoZ: 0,
        giroPinzaZ: 0,
        aperturaPinza: 15,
        reiniciar: function () {
            angulo = 0
            location.reload();
        },
        color: "rgb(255,0,0)"
    }
    var gui = new dat.GUI();
    var sub = gui.addFolder("Controles Robot")
    sub.add(effectControl, "giroBase", -180, 180, 1).name("Giro Base");
    sub.add(effectControl, "giroBrazo", -45, 45, 1).name("Giro Brazo");
    sub.add(effectControl, "giroAntebrazoZ", -180, 180, 1).name("Giro Antebrazo Z");
    sub.add(effectControl, "giroAntebrazoY", -90, 90, 1).name("Giro Antebrazo Y");
    sub.add(effectControl, "giroPinzaZ", -40, 220, 1).name("Giro Pinza");
    sub.add(effectControl, "aperturaPinza", 0, 15, 1).name("Separacion pinza");

    sub.add(effectControl, "reiniciar")
    var sensorColor = sub.addColor(effectControl, "color").name("Color")
    sensorColor.onChange(function (color) {
        robot.traverse(function (hijo) {
            if (hijo instanceof THREE.Mesh) {
                hijo.material.color = new THREE.Color(color)
            }
        })
    })


}