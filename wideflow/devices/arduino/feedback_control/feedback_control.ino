/*
  feedback control
  control feedback devices by recieving inputs from a serial port
  
  * Serial communication functions were taken from:
      "Demo of PC-Arduino comms using Python"
  by Robin2
  https://forum.arduino.cc/t/demo-of-pc-arduino-comms-using-python/219184
  
*/


//==========================================================
// set communication variables =============================
const byte buffSize = 3;
char inputBuffer[buffSize];
const char startMarker = '<';
const char endMarker = '>';
byte bytesRecvd = 0;
boolean readInProgress = false;
boolean newDataFromPC = false;
boolean allReceived = false;

char messageFromPC[buffSize] = {0};
const char activateFeedback = 'H';
const char sendReport = 'R';
const char openValve = 'V';
const char closeValve = 'v';

char dataSend;
byte msg;

//==========================================================
// set arduino variables ===================================
const int valvePin = 48; // the pin that the solenoid is attached to
const int lickPortPin = 52; //the pin that the lick port is attached to
const int ledPin = 46; // the pin that the LED is attached to
const int speakerPin = 44; // the pin that the speaker is attached to

int ledAnalogVal = 100; // control the LED illumination intensity
int speakerAnalogVal = 8;

byte lickPortStat;

//==========================================================
// set feedback control variables ==========================
unsigned long globalClock;
unsigned long valveClock = 0;
unsigned long ledClock = 0;
unsigned long speakerClock = 0;
unsigned long speakerActivationClock = 0;
int valveActivationTime = 14;
int ledActivationTime = 1500;
int speakerActivationTime = 500;
int speakerDelayTime = 5000;


//==========================================================
//==========================================================
//==========================================================
//==========================================================
void setup() {
  // initialize serial communication:
  Serial.begin(9600);

  // initialize the LED pin as an output:
  pinMode(ledPin, OUTPUT);
  // initialize the valve pin as an output:
  pinMode(valvePin, OUTPUT);
  // initialize the TTL pin as an output:
  pinMode(lickPortPin, INPUT);
  // initialize the speaker pin as an output:
  pinMode(speakerPin, OUTPUT);

  // tell the PC we are ready
  //Serial.println("<Arduino is ready>");

}


//==========================================================
//==========================================================
void loop() {
  globalClock = millis();
  lickPortStat = digitalRead(lickPortPin);
  getDataFromPC();
  process();

}


//==========================================================
//==========================================================
void process() {
    if (allReceived) {
      msg = inputBuffer[0];
      if (msg == activateFeedback){   // open the valve and light the LED
        valveClock = millis();
        ledClock = millis();
        speakerClock = millis();
        digitalWrite(valvePin, HIGH);
        analogWrite(ledPin, ledAnalogVal);
      }

      else if (msg == sendReport){
        if (lickPortStat == 0) {
          dataSend = '0';
        }
        else{
          dataSend = '1';
        }
        sendDataToPC();
      }

      else if (msg == openValve){
        digitalWrite(valvePin, HIGH);
      }

      else if (msg == closeValve){
        digitalWrite(valvePin, LOW);
      }
      
      allReceived = false; 
  }
  
  // if valveActivationTime passed since valve has been opened - close it
  if ((valveClock > 0) && (globalClock > (valveClock + valveActivationTime))) {
    valveClock = 0;
    digitalWrite(valvePin, LOW);
  }

  // if ledActivationTime passed since led has been lit - close it
  if ((ledClock > 0) && (globalClock > (ledClock + ledActivationTime))) {
    ledClock = 0;
    digitalWrite(ledPin, LOW);
  }
  
  // if lickPort is connected while no reward has been given in the last "speakerDelayTime" - play aversive sound
  if ((globalClock > (speakerClock + speakerDelayTime)) && lickPortStat == 0){
    analogWrite(speakerPin, speakerAnalogVal);
    speakerActivationClock = millis();
  }
  
  // turn off aversive sound if speakerActivationTime has passed
  else if (globalClock > (speakerActivationClock + speakerActivationTime)) {
    analogWrite(speakerPin, 0);
  }
  

}


//==========================================================
//==========================================================
void getDataFromPC() {

  // receive data from PC and save it into inputBuffer

  if(Serial.available() > 0) {

    char x = Serial.read();
    if (x == endMarker) {
      allReceived = true;
      readInProgress = false;
      inputBuffer[bytesRecvd] = 0;
    }
    
    if(readInProgress) {
      inputBuffer[bytesRecvd] = x;
      bytesRecvd ++;
      if (bytesRecvd == buffSize) {
        bytesRecvd = buffSize - 1;
      }
    }

    if (x == startMarker) { 
      bytesRecvd = 0; 
      readInProgress = true;
    }
  }
}


//==========================================================
//==========================================================
void sendDataToPC() {
  Serial.print(startMarker);
  Serial.print(dataSend);
  Serial.println(endMarker);

}
