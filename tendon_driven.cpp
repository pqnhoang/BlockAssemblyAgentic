#include <SimpleFOC.h>

MagneticSensorSPI sensor = MagneticSensorSPI(AS5147_SPI, 53);
BLDCMotor motor = BLDCMotor(11);
BLDCDriver3PWM driver = BLDCDriver3PWM(9, 5, 6, 8);

float target_angle = 0;
float amplitude = 0.5;
float frequency = 0.2;
unsigned long previousMillis = 0;
const long interval = 10;
bool RunModeSineWave = true;
bool RunModeCustomPosition = false;
float sineValue = 0;

void calibration()
{
  if (Serial.available())
  {
    char cmd = Serial.read();
    switch (cmd)
    {
    case 't':
      target_angle = Serial.parseFloat();
      break;
    case 'a':
      amplitude = Serial.parseFloat();
      break;
    case 'f':
      frequency = Serial.parseFloat();
      break;
    case 'n':
      RunModeSineWave = true;
      RunModeCustomPosition = false;
      break;
    case 'm':
      RunModeSineWave = false;
      RunModeCustomPosition = true;
      break;
    case 'e':
      motor.enable();
      break;
    case 's':
      motor.disable();
      break;
    }
  }
}

void setup()
{
  Serial.begin(115200);

  sensor.init();
  motor.linkSensor(&sensor);

  driver.voltage_power_supply = 12;
  driver.init();

  motor.linkDriver(&driver);

  motor.foc_modulation = FOCModulationType::SpaceVectorPWM;
  motor.torque_controller = TorqueControlType::voltage;
  motor.controller = MotionControlType::angle;

  motor.PID_velocity.P = 0.07;
  motor.PID_velocity.I = 5;
  motor.PID_velocity.D = 0.001;
  motor.LPF_velocity.Tf = 0.01;
  motor.PID_velocity.output_ramp = 6.0;

  motor.P_angle.P = 10;
  motor.LPF_angle.Tf = 0.01;

  motor.velocity_limit = 5.0;

  motor.sensor_direction = Direction::CCW;
  motor.sensor_offset = 0.00;
  motor.zero_electric_angle = 3.05;

  motor.init();
  motor.initFOC();
}

void loop()
{
  calibration();
  motor.loopFOC();

  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval)
  {
    previousMillis = currentMillis;
    float phase = 2 * PI * frequency * (currentMillis / 1000.0);
    sineValue = amplitude * sin(phase);
  }

  if (RunModeSineWave)
    motor.move(sineValue);
  else if (RunModeCustomPosition)
    motor.move(target_angle);
}
