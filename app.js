var net = require('net');
var k = "2";
var beacon_id = '0';
var Rssi = '0';
var exec = require('child_process').exec;
var arg1 = 'hello';
var arg2 = 'world';
var filename = 'Hashing_KNN.py'
//var filename = 'test.py'
//test

/*
exec('python' + ' ' + filename + ' ' + arg1 + ' ' + arg2, function (err, stdout, stderr) {
  if (err) {
    console.log('stderr', err);
  }
  if (stdout) {
    console.log('output', stdout);
  }

});
*/


function full(Array) {
  for (let i = 0; i < Array.length; i++) {
    if (Array[i] == '0')
      return false;
  }
  return true;
}
let buffer = new Array(6);
buffer = ['0', '0', '0', '0', '0', '0'];
var Beacon_RSSi = '';
var t = 0;
var beacon_int = 0
var Rssi_int = 0;
var toclient = '0';


var count = 0;
var clientHandler = function (socket) {
  //客戶端傳送資料的時候觸發data事件
  socket.on('data', function dataHandler(data) {//data是客戶端傳送給伺服器的資料

    //console.log(socket.remoteAddress, socket.remotePort, 'send', data.toString());
    k = data.toString();
    var K = k.split(",", 2);
    beacon_id = K[0];
    Rssi = K[1];
    //console.log("zz" + beacon_id + "zz");
    let numStr = beacon_id.replace(/[^0-9]/ig, "");
    //console.log(Rssi.length);
    beacon_int = parseInt(numStr);
    Rssi_int = parseInt(Rssi);
    console.log(beacon_int, Rssi_int);
    Rssi_int = (Rssi_int + 101) / (-46 + 101);
    if (isNaN(Rssi_int)) {
      console.log(beacon_int);
      console.log("NAN");
    }
    if (beacon_int == 7) {
      buffer[5] = Rssi_int;
    }//因為用1 2 3 4 5 7
    else
      buffer[beacon_int - 1] = Rssi_int;

    //console.log(Beacon_RSSi);
    if (full(buffer)) {
      //count++;
      Beacon_RSSi = `{q"Beacon_1q":q"${buffer[0]}q",q"Beacon_2q":q"${buffer[1]}q",q"Beacon_3q":q"${buffer[2]}q",q"Beacon_4q":q"${buffer[3]}q",q"Beacon_5q":q"${buffer[4]}q",q"Beacon_7q":q"${buffer[5]}q"}`;
      //console.log(Beacon_RSSi);
      //console.log(count);
      //if (count == 6) {
      exec('python' + ' ' + filename + ' ' + Beacon_RSSi + '' + count, function (err, stdout, stderr) {
        if (err) {
          console.log('stderr', err);
        }
        if (stdout) {
          console.log('output', stdout);
          toclient = stdout;
          //socket.write(toclient);
        }
        //buffer = ['0', '0', '0', '0', '0', '0'];
        //count = 0;
      });
      buffer = ['0', '0', '0', '0', '0', '0'];
      //count = 0;

      //}
      count++;
    }


    //console.log("beacon_id:", beacon_id, "Rssi:", Rssi)
    //客戶向server傳送訊息
    socket.write(toclient);
    //console.log("k");
    //socket.write('server received\n');
  });

  //當對方的連線斷開以後的事件
  socket.on('close', function () {
    //console.log(socket.remoteAddress, socket.remotePort, 'disconnected');
  })
  socket.on('error', (err) => {
    console.log(err);
  })
};

var app = net.createServer(clientHandler);
app.listen(8000, '140.116.72.75');
console.log('tcp server running on tcp://', '140.116.72.75', ':', 8000);

