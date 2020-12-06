/**
 * Proyecto final de GPC basado en fisicas.js de rvivo (http://personales.upv.es/rvivo).
 * @requires three_r96.js, coordinates.js, orbitControls.js, cannon.js, tween.js, stats_r16.js
 * @date 2020
 */

// Globales convenidas por threejs
var renderer, scene, camera;
// Control de camara
var cameraControls;
// Monitor de recursos
var stats;

// Mundo fisico
var world, reloj;
// Objetos
var objectos = []; // 3D Models
var ground;
var airplane;
// Colores
//COLORS
var Colors = {
    red: 0xf25346,
    white: 0xd8d0d1,
    brown: 0x59332e,
    pink: 0xF5986E,
    brownDark: 0x23190f,
    blue: 0x68c3c0,
};

/**
 * Construye una bola con cuerpo y vista
 */
/// FUNCIONES CREAR OBJECTOS
function esfera(radio, posicion, material) {
    var masa = 1;
    this.body = new CANNON.Body({
        mass: masa,
        material: material
    });
    this.body.addShape(new CANNON.Sphere(radio));
    this.body.position.copy(posicion);
    this.visual = new THREE.Mesh(new THREE.SphereGeometry(radio),
        new THREE.MeshBasicMaterial({
            wireframe: true
        }));
    this.visual.position.copy(this.body.position);
}

function Plane(material, position) {
    // this.mesh = new THREE.Object3D();
    // this.mesh.name = "airPlane";
    var mat = new THREE.MeshPhongMaterial({
        color: Colors.blue,
        // transparent:true,
        // opacity:.6,
        shading: THREE.FlatShading,
    });
    this.mesh = new THREE.Mesh(new THREE.SphereGeometry(25),
        mat);
    // Cannon
    var masa = 1;
    this.body = new CANNON.Body({
        mass: masa,
        material: material
    });
    this.body.addShape(new CANNON.Sphere(5));
    this.body.position.copy(position);
    this.visual = this.mesh;
    this.visual.position.copy(this.body.position);

};

function Sky(material, position) {
    this.mesh = new THREE.Object3D();
    this.nClouds = 20;
    this.clouds = [];
    var stepAngle = Math.PI * 2 / this.nClouds;
    for (var i = 0; i < this.nClouds; i++) {
        var c = new Cloud();
        this.clouds.push(c);
        var a = stepAngle * i;
        var h = 750 + Math.random() * 200;
        c.mesh.position.y = Math.sin(a) * h;
        c.mesh.position.x = Math.cos(a) * h;
        c.mesh.position.z = -400 - Math.random() * 400;
        c.mesh.rotation.z = a + Math.PI / 2;
        var s = 1 + Math.random() * 2;
        c.mesh.scale.set(s, s, s);
        this.mesh.add(c.mesh);
    }
    // this.mesh.rotation.x =  * Math.PI / 180
    var masa = 0;
    this.body = new CANNON.Body({
        mass: masa,
        material: material
    });
    this.body.addShape(new CANNON.Sphere(1000));
    this.body.position.copy(position);
    this.visual = this.mesh;
    this.visual.position.copy(this.body.position);
}

function Ground(material, position) {
    var textureLoader = new THREE.TextureLoader();
    var map = textureLoader.load('./textures/grass.jpg');
    // Cannon
    var masa = 1;
    this.body = new CANNON.Body({
        mass: masa,
        material: material
    });
    this.body.addShape(new CANNON.Box(new CANNON.Vec3(100, 100, 10)));
    this.body.position.copy(position);
    // Visuales
    var geom = new THREE.BoxGeometry(1000, 1000, 10, 40, 10);
    // geom.applyMatrix(new THREE.Matrix4().makeRotationX(-Math.PI / 2));
    var mat = new THREE.MeshPhongMaterial({
        color: Colors.blue,
        map: map,
        // transparent:true,
        // opacity:.6,
        shading: THREE.FlatShading,
    });
    mesh = new THREE.Mesh(geom, mat);
    mesh.receiveShadow = true;
    this.visual = mesh;
    this.visual.position.copy(this.body.position);
}

// From https://github.com/yakudoo/TheAviator
Cloud = function () {
    var textureLoader = new THREE.TextureLoader();
    var map = textureLoader.load('./textures/cloud.jpg');
    this.mesh = new THREE.Object3D();
    this.mesh.name = "cloud";
    var geom = new THREE.CubeGeometry(20, 20, 20);
    var mat = new THREE.MeshPhongMaterial({
        color: Colors.white,
        map: map,

    });

    var nBlocs = 3 + Math.floor(Math.random() * 3);
    for (var i = 0; i < nBlocs; i++) {
        var m = new THREE.Mesh(geom.clone(), mat);
        m.position.x = i * 15;
        m.position.y = Math.random() * 10;
        m.position.z = Math.random() * 10;
        m.rotation.z = Math.random() * Math.PI * 2;
        m.rotation.y = Math.random() * Math.PI * 2;
        var s = .1 + Math.random() * .9;
        m.scale.set(s, s, s);
        m.castShadow = true;
        m.receiveShadow = true;
        this.mesh.add(m);
    }
}
/**
 * Inicializa las luces
 */
var ambientLight, hemisphereLight, shadowLight;

function createLights() {

    hemisphereLight = new THREE.HemisphereLight(0xaaaaaa, 0x000000, .9)
    shadowLight = new THREE.DirectionalLight(0xffffff, .9);
    shadowLight.position.set(150, 350, 350);
    shadowLight.castShadow = true;
    shadowLight.shadow.camera.left = -400;
    shadowLight.shadow.camera.right = 400;
    shadowLight.shadow.camera.top = 400;
    shadowLight.shadow.camera.bottom = -400;
    shadowLight.shadow.camera.near = 1;
    shadowLight.shadow.camera.far = 1000;
    shadowLight.shadow.mapSize.width = 2048;
    shadowLight.shadow.mapSize.height = 2048;

    scene.add(hemisphereLight);
    scene.add(shadowLight);
}
/**
 * Inicializa el mundo fisico con un
 * suelo y cuatro paredes de altura infinita
 */
function initPhysicWorld() {
    // Mundo
    world = new CANNON.World();
    world.gravity.set(0, 0, 0);
    ///world.broadphase = new CANNON.NaiveBroadphase();
    world.solver.iterations = 10;

    // Material y comportamiento
    var groundMaterial = new CANNON.Material("groundMaterial");
    var planeMaterial = new CANNON.Material("planeMaterial");
    var skyMaterial = new CANNON.Material("skyMaterial");
    world.addMaterial(planeMaterial);
    world.addMaterial(skyMaterial);
    world.addMaterial(groundMaterial);
    // -existe un defaultContactMaterial con valores de restitucion y friccion por defecto
    // -en caso que el material tenga su friccion y restitucion positivas, estas prevalecen
    var planeGroundContactMaterial = new CANNON.ContactMaterial(groundMaterial, planeMaterial, {
        friction: 0.3,
        restitution: 0.7
    });
    world.addContactMaterial(planeGroundContactMaterial);

    // Suelo
    var groundShape = new CANNON.Plane();
    var ground = new CANNON.Body({
        mass: 0,
        material: groundMaterial
    });
    ground.addShape(groundShape);
    ground.quaternion.setFromAxisAngle(new CANNON.Vec3(1, 0, 0), -Math.PI / 2);
    world.addBody(ground);
}

/**
 * Inicializa la escena visual
 */
function initVisualWorld() {
    // Inicializar el motor de render
    HEIGHT = window.innerHeight;
    WIDTH = window.innerWidth;
    renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: true
    });
    renderer.setSize(WIDTH, HEIGHT);
    renderer.shadowMap.enabled = true;
    container = document.getElementById('container');
    container.appendChild(renderer.domElement);

    // Crear el grafo de escena
    scene = new THREE.Scene();

    // Reloj
    reloj = new THREE.Clock();
    reloj.start();

    // Crear y situar la camara
    aspectRatio = WIDTH / HEIGHT;
    fieldOfView = 60;
    nearPlane = 1;
    farPlane = 10000;
    camera = new THREE.PerspectiveCamera(
        fieldOfView,
        aspectRatio,
        nearPlane,
        farPlane
    );
    camera.position.x = 0;
    camera.position.z = 500;
    camera.position.y = 100;
    // Control de camara
    cameraControls = new THREE.OrbitControls(camera, renderer.domElement);
    cameraControls.target.set(0, 0, 0);

    // STATS --> stats.update() en update()
    stats = new Stats();
    stats.showPanel(0); // FPS inicialmente. Picar para cambiar panel.
    document.getElementById('container').appendChild(stats.domElement);

    // Callbacks
    window.addEventListener('resize', updateAspectRatio);
}

/**
 * Carga los objetos es el mundo físico y visual
 */
function loadWorld() {
    // Genera las esferas
    //Buscamos materiales
    var groundMaterial;
    var planeMaterial;
    var skyMaterial;
    for (i = 0; i < world.materials.length; i++) {
        if (world.materials[i].name === "groundMaterial") groundMaterial = world.materials[i];
        if (world.materials[i].name === "planeMaterial") planeMaterial = world.materials[i];
        if (world.materials[i].name === "skyMaterial") skyMaterial = world.materials[i];
    }
    // Creamos el avion
    airplane = new Plane(planeMaterial, new CANNON.Vec3(0, 10, 200));
    world.addBody(airplane.body);
    scene.add(airplane.visual);
    objectos.push(airplane)
    // Creamos el suelo
    ground = new Ground(groundMaterial, new CANNON.Vec3(0, 0, 0));
    world.addBody(ground.body);
    scene.add(ground.visual);
    objectos.push(ground)
    // Creamos el cielo
    sky = new Sky(skyMaterial, new CANNON.Vec3(0, -500, -0));
    world.addBody(sky.body);
    scene.add(sky.visual);
    objectos.push(sky)

    //Giramos el cielo
    var giro = new TWEEN.Tween(sky.visual.rotation).to({
        x: 0,
        y: 0,
        z: 2 * Math.PI,
    }, 10000);
    giro.repeat(Infinity);
    giro.start();
    createLights();
    scene.add(new THREE.AxisHelper(5));

    // world.addContactMaterial(materialPlane);
    // world.addContactMaterial(materialSky);
    // world.addContactMaterial(materialPlane);
}

/**
 * Isotropía frente a redimension del canvas
 */
function updateAspectRatio() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
}

/**
 * Actualizacion segun pasa el tiempo
 */
function update() {
    var segundos = reloj.getDelta(); // tiempo en segundos que ha pasado
    world.step(segundos); // recalcula el mundo tras ese tiempo

    for (var i = 0; i < objectos.length; i++) {
        objectos[i].visual.position.copy(objectos[i].body.position);
        objectos[i].visual.quaternion.copy(objectos[i].body.quaternion);
    };

    // Actualiza el monitor
    stats.update();
    // Actualiza el movimeinto de las nubes
    TWEEN.update();
}

/**
 * Update & render
 */
function render() {
    requestAnimationFrame(render);
    update();
    renderer.render(scene, camera);
}

function addcubes() {
    mesh = new THREE.Object3D();
    nCubes = 5;
    cubes = [];
    var stepAngle = Math.PI * 2 / this.nClouds;
    for (var i = 0; i < nCubes; i++) {
        var c = new Cloud();
        cubes.push(c);
        var a = stepAngle * i;
        var h = 750 + Math.random() * 200;
        c.mesh.position.y = Math.sin(a) * h;
        c.mesh.position.x = Math.cos(a) * h;
        c.mesh.position.z = -400 - Math.random() * 400;
        c.mesh.rotation.z = a + Math.PI / 2;
        var s = 1 + Math.random() * 2;
        c.mesh.scale.set(s, s, s);
        mesh.add(c.mesh);
    }
    world.addBody(sky.body);
    scene.add(mesh);
    console.log("cubeando")
}

function loop() {
    ground.body.position.x += 1;
    ground.visual.position.copy(ground.body.position);
    update();
    requestAnimationFrame(loop);
}

function init(event) {
    //   document.addEventListener('mousemove', handleMouseMove, false);
    initPhysicWorld();
    initVisualWorld();
    loadWorld();
    render();
    loop();
    window.addEventListener('keydown', function movekey(event) {
        if (event.keyCode == 39 || event.keyCode == 68) {
            airplane.body.position.x += 10;
            airplane.visual.position.copy(airplane.body.position);
            camera.position.x += 10;
            sky.body.position.x += 10;
            sky.visual.position.copy(sky.body.position);
            // Luces
            // shadowLight.position.x += 10;
        } else if ((event.keyCode == 38 || event.keyCode == 87)) {
            airplane.body.position.y += 10;
            airplane.visual.position.copy(airplane.body.position);
            console.log("w")
        } else if ((event.keyCode == 37 || event.keyCode == 65)) {
            airplane.body.applyImpulse(new CANNON.Vec3(0, -10, 0), airplane.body.position)
            airplane.visual.position.copy(airplane.body.position);
            // camera.position.x -= 10;
            sky.body.position.x -= 10;
            sky.visual.position.copy(sky.body.position); // Luces
            // shadowLight.position.x -= 10;
            console.log("a")

        } else if ((event.keyCode == 40 || event.keyCode == 83)) {
            airplane.body.position.y -= 10;
            airplane.visual.position.copy(airplane.body.position);
            console.log("s")
        }
    }, false);
}


window.addEventListener('load', init, false);