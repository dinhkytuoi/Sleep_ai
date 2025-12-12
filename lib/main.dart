import 'package:alarm_app/services/alarm_service.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'screens/logo_screen.dart';
import 'screens/question_screen.dart';
import 'screens/question_age_screen.dart';
import 'screens/home_screen.dart';
import 'screens/profile_screen.dart';
import 'screens/alarm_screen.dart';
import 'screens/sound_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/alarm_ring_screen.dart';

final GlobalKey<NavigatorState> appNavigatorKey = GlobalKey<NavigatorState>();
void main() async {
  WidgetsFlutterBinding.ensureInitialized(); // 🔹 Bắt buộc cho async

  await AlarmService.init(); // 🔹 Khởi tạo plugin báo thức

  // 🔹 Kiểm tra xem người dùng đã làm khảo sát chưa
  final prefs = await SharedPreferences.getInstance();
  final seenOnboarding = prefs.getBool('seenOnboarding') ?? false;

  runApp(MyApp(initialRoute: seenOnboarding ? '/home' : '/logo'));
}

class MyApp extends StatelessWidget {
  final String initialRoute; // 👈 Thêm dòng này
  const MyApp({super.key, required this.initialRoute});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cycle Alarm',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      navigatorKey: appNavigatorKey,
      initialRoute: initialRoute,
      routes: {
        '/logo': (_) => const LogoScreen(),
        '/question': (_) => const QuestionScreen(),
        '/question_age': (_) => const QuestionAgeScreen(),
        '/home': (_) => const HomeScreen(),
        '/profile': (_) => const ProfileScreen(),
        '/sound': (_) => const SoundScreen(),
        '/settings': (_) => const SettingsScreen(),
        //'/alarm_ring': (_) => const AlarmRingScreen(),
        '/alarm_ring': (context) {
          final args = ModalRoute.of(context)?.settings.arguments;
          return AlarmRingScreen(
            toneAsset: args is String && args.isNotEmpty
                ? args
                : 'assets/audio/drizzling.mp3',
          );
        },
      },
      onGenerateRoute: (settings) {
        if (settings.name == '/alarm') {
          final String alarmTime = (settings.arguments is String &&
                  (settings.arguments as String).isNotEmpty)
              ? settings.arguments as String
              : '11:25 PM';
          return MaterialPageRoute(
            builder: (_) => AlarmScreen(
              alarmTime: alarmTime,
              enabled: false,
              onToggle: (bool v) {},
            ),
            settings: settings,
          );
        }
        return null;
      },
    );
  }
}
