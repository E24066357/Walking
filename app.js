var net = require('net');
var k = "2";
var beacon_id = '0';
var Rssi = '0';
var exec = require('child_process').exec;
var arg1 = 'hello';
var arg2 = 'world';
var filename = 'Hashing_KNN.py'
//test


exec('python' + ' ' + filename + ' ' + arg1 + ' ' + arg2, function (err, stdout, stderr) {
    if (err) {
        console.log('stderr', err);
    }
    if (stdout) {
        console.log('output', stdout);
    }

});


function full(Array) {
    for (let i = 0; i < Array.length; i++) {
        if (!buffer[i])
            return false
    }
    return true
}
/*
let buffer = new Array(6);
buffer[beacon_id] = Rssi
for (let i = 0; i < buffer.length; i++) {
    buffer[i] = 2;
}
if (full(buffer)) {
    Beacon_RSSi = `{"Beacon_1":"${buffer[0]}","Beacon_2":"${buffer[1]}","Beacon_3":"${buffer[2]}","Beacon_4":"${buffer[3]}","Beacon_5":"${buffer[4]}","Beacon_6":"${buffer[5]}"}`
    console.log(Beacon_RSSi)
}
*/

/*
var clientHandler = function (socket) {
    //客戶端傳送資料的時候觸發data事件
    socket.on('data', function dataHandler(data) {//data是客戶端傳送給伺服器的資料
        console.log(socket.remoteAddress, socket.remotePort, 'send', data.toString());
        k = data.toString();
        var K = k.split(",", 2);
        beacon_id = K[0];
        Rssi = K[1];
        console.log("beacon_id:", beacon_id, "Rssi:", Rssi)
        //客戶向server傳送訊息
        socket.write('server received\n');
    });

    //當對方的連線斷開以後的事件
    socket.on('close', function () {
        console.log(socket.remoteAddress, socket.remotePort, 'disconnected');
    })
};

var app = net.createServer(clientHandler);
app.listen(8000, '140.116.72.75');
console.log('tcp server running on tcp://', '140.116.72.75', ':', 8000);
console.log(k)
*/