/*
  feedback control
  control a device by recieving inputs from a serial port

*/

const int valvePin = 42; // the pin that the solenoid is attached to
const int lickPortPin = 52; //the pin that the lick port is attached to
const int ledPin = 46; // the pin that the LED is attached to

int incomingByte;      // a variable to read incoming serial data info
int lickPortStat;

int valveClock = 0;
int valveActivationTime = 1000;
int ledActivationTime = 2000;
int ledClock = 0;
int globalClock;

void setup() {
  // initialize serial communication:
  Serial.begin(9600);
  
  // initialize the TTL pin as an output:
  pinMode(ledPin, OUTPUT);
  // initialize the valve pin as an output:
  pinMode(valvePin, OUTPUT);
  // initialize the TTL pin as an output:
  pinMode(lickPortPin, INPUT);
  
  //digitalWrite(ledPin, HIGH);
}

void loop() {
  // see if there's incoming serial data:
  if (Serial.available() > 0) {
    // read the oldest byte in the serial buffer:
    incomingByte = Serial.read();

    // if it's a capital T, set vavlePin to HIGH - open selonoid valve:
    if (incomingByte == 'H') {
      valveClock = millis();
      ledClock = millis();
      digitalWrite(valvePin, HIGH);
      digitalWrite(ledPin, HIGH);
    }
  }

  // if valveActivationTime passed since valve has been opened - close it
  globalClock = millis();
  if ((valveClock > 0) && (globalClock > (valveClock + valveActivationTime))) {
    valveClock = 0;
    digitalWrite(valvePin, LOW);
    Serial.println("closing valve");
  }
  
  // if ledActivationTime passed since led has been lit - close it
  if ((ledClock > 0) && (globalClock > (ledClock + ledActivationTime))) {
    ledClock = 0;
    digitalWrite(ledPin, LOW);
    Serial.println("closing led");
  }

  // check lickPort status
  lickPortStat = digitalRead(lickPortPin);
  //Serial.println(lickPortStat);
}
