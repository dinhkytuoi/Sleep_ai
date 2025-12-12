import 'package:shared_preferences/shared_preferences.dart';

/// Lưu/local profile + credential đơn giản bằng SharedPreferences
class UserPrefs {
  // ---------- Auth ----------
  static Future<void> saveEmail(String email) async {
    final p = await SharedPreferences.getInstance();
    await p.setString('email', email);
  }

  static Future<void> savePassword(String password) async {
    final p = await SharedPreferences.getInstance();
    await p.setString('password', password);
  }

  static Future<String?> getEmail() async {
    final p = await SharedPreferences.getInstance();
    return p.getString('email');
  }

  static Future<String?> getPassword() async {
    final p = await SharedPreferences.getInstance();
    return p.getString('password');
  }

  // ---------- Profile ----------
  static Future<void> saveName(String name) async {
    final p = await SharedPreferences.getInstance();
    await p.setString('name', name);
  }

  static Future<void> saveAge(String age) async {
    final p = await SharedPreferences.getInstance();
    await p.setString('age', age);
  }

  static Future<void> saveGender(String gender) async {
    final p = await SharedPreferences.getInstance();
    await p.setString('gender', gender);
  }

  static Future<String?> getName() async {
    final p = await SharedPreferences.getInstance();
    return p.getString('name');
  }

  static Future<String?> getAge() async {
    final p = await SharedPreferences.getInstance();
    return p.getString('age');
  }

  static Future<String?> getGender() async {
    final p = await SharedPreferences.getInstance();
    return p.getString('gender');
  }

  static Future<void> clearAll() async {
    final p = await SharedPreferences.getInstance();
    await p.clear();
  }
}
