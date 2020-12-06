//Seminario #1 GPC


//Shader vertices
var VSHADER_SOURCE =
    'attribute vec4 posicion;       \n' +
    'void main(){                   \n' +
    '  gl_Position = posicion;      \n' +
    '  gl_PointSize = 10.0;         \n' +
    '}       \n';


var FSHADER_SOURCE =
    'void main(){                   \n' +
    '  gl_FragColor = vec4(1.0,0.0,0.0,1.0);      \n' +
    '}                              \n';


function main() {
    //Recuperar el cavas
    var canvas = document.getElementById("canvasJaime");
    if (!canvas) {
        console.log("Fallo de carga del canvas");
        return;
    }

    // Recuperar el contexto de render
    var gl = getWebGLContext(canvas)
    if (!gl) {
        console.log("Fallo de carga del contexto del render");
        return;
    }

    // Cargar, compilar y montar los shaders en un 'program'
    if (!initShaders(gl, VSHADER_SOURCE, FSHADER_SOURCE)) {
        console.log("Se ha liado en la carga de los shaders")
        return;
    }

    gl.clearColor(0.0, 0.0, 0.3, 1.0);
    gl.clear(gl.COLOR_BUGGET_BIT);
    var coordenadas = gl.getAttribLocation(gl.program, 'posicion');

    canvas.onmousedown = function (evento) {
        click(evento, gl, canvas, coordenadas)
    }
}

var puntos = []; //Array de puntos
function click(evento, gl, canvas, coordenadas) {
    // p.dot = d.vector*[x,y] = g.vector*[x',y'] = g.vector*A*[x,y] Que es A?
    var x = evento.clientX;
    var y = evento.clientY;
    var rect = evento.target.getBoundingClientRect();

    //Conversion de cordenadas
    x = ((x - rect.left) - canvas.width / 2) * 2 / canvas.width;
    y = (canvas.height / 2 - (y - rect.top)) * 2 / canvas.height;

    //Guardar los puntos
    puntos.push(x);
    puntos.push(y);

    //Borrar el canvas
    gl.clear(gl.COLOR_BUGGET_BIT);

    //Inserta las coodenadas de los puntos como atributos y los dibuja uno a uno

    for (var i = 0; i < puntos.length; i += 2) {
        gl.vertexAttrib3f(coordenadas, puntos[i], puntos[i + 1], 0.0);
        gl.drawArrays(gl.POINTS, 0, 1);
    }

}