#include <LiquidCrystal.h>

LiquidCrystal lcd(7, 6, 5, 4, 3, 2);

const int buzzerPin = 9;
const int detectPin = 8;

void setup() {
  pinMode(buzzerPin, OUTPUT);
  pinMode(detectPin, INPUT_PULLUP);

  lcd.begin(16, 2);
  lcd.print("RF Monitor");
  delay(1500);
  lcd.clear();
}

void loop() {
  int detection = digitalRead(detectPin);

  if (detection == LOW) {
    digitalWrite(buzzerPin, HIGH);
    lcd.setCursor(0, 0);
    lcd.print("DRONE ALERT!  ");
    lcd.setCursor(0, 1);
    lcd.print("RF ACTIVITY   ");
  } else {
    digitalWrite(buzzerPin, HIGH);
    lcd.setCursor(0, 0);
    lcd.print("AREA CLEAR    ");
    lcd.setCursor(0, 1);
    lcd.print("Monitoring... ");
  }

  delay(300);
}
