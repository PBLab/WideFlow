/*
  feedback control
  control a device by recieving inputs from a serial port

*/

const int valvePin = 42; // the pin that the solenoid is attached to
const int lickPortPin = 52; //the pin that the lick port is attached to
const int ledPin = 46; // the pin that the LED is attached to

int incomingByte;      // a variable to read incoming serial data info
int lickPortStat;

double valveClock = 0;
int valveActivationTime = 1000;
int ledActivationTime = 2000;
double ledClock = 0;
double globalClock;

int ledAnalogVal = 100; // control the LED illumination intensity

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

    if (incomingByte == 'H') {
      valveClock = millis();
      ledClock = millis();
      digitalWrite(valvePin, HIGH);
      analogWrite(ledPin, ledAnalogVal);
    }
  }

  // if valveActivationTime passed since valve has been opened - close it
  globalClock = millis();
  if ((valveClock > 0) && (globalClock > (valveClock + valveActivationTime))) {
    valveClock = 0;
    digitalWrite(valvePin, LOW);
  }

  // if ledActivationTime passed since led has been lit - close it
  if ((ledClock > 0) && (globalClock > (ledClock + ledActivationTime))) {
    ledClock = 0;
    digitalWrite(ledPin, LOW);
  }

  // check lickPort status
  lickPortStat = digitalRead(lickPortPin);
  Serial.print(lickPortStat);
  //Serial.println(lickPortStat);
  //Serial.write(lickPortStat);
}
