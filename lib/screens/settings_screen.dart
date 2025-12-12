// lib/screens/settings_screen.dart
import 'dart:ui';
import 'package:flutter/material.dart';

// import các màn detail
import 'general_settings_screen.dart';
import 'faq_screen.dart';
import 'feedback_screen.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  final double _cardRadius = 18;

  Widget _buildRowItem({
    required IconData icon,
    required String title,
    String? subtitle,
    Widget? trailing,
    VoidCallback? onTap,
  }) {
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 12.0, horizontal: 6),
        child: Row(
          children: [
            Container(
              width: 46,
              height: 46,
              decoration: BoxDecoration(
                color: Colors.white12,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, size: 24, color: Colors.white),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title,
                      style: const TextStyle(
                          fontSize: 16, fontWeight: FontWeight.w600)),
                  if (subtitle != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 4.0),
                      child: Text(subtitle,
                          style: const TextStyle(
                              fontSize: 13, color: Colors.white70)),
                    ),
                ],
              ),
            ),
            if (trailing != null) trailing,
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final bottomSafe = MediaQuery.of(context).padding.bottom;
    final theme = Theme.of(context);

    return Scaffold(
      extendBody: true,
      body: Container(
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage("assets/images/background.png"),
            fit: BoxFit.cover,
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding:
                const EdgeInsets.symmetric(horizontal: 18.0, vertical: 18.0),
            child: Column(
              children: [
                // Header
                Row(
                  children: [
                    Expanded(
                      child: Text('ME',
                          style: theme.textTheme.headlineSmall?.copyWith(
                            fontWeight: FontWeight.bold,
                            fontSize: 32,
                            color: Colors.white,
                          )),
                    ),
                    // Avatar: nhấn vào sẽ mở Profile (named route '/profile')
                    GestureDetector(
                      onTap: () {
                        Navigator.pushNamed(context, '/profile');
                      },
                      child: const CircleAvatar(
                        radius: 20,
                        backgroundImage: AssetImage("assets/images/avatar.png"),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 18),

                // Big card (Settings) - note: Sleep Settings removed
                Expanded(
                  child: SingleChildScrollView(
                    child: Column(
                      children: [
                        // Card: Settings (primary) - Sleep removed, General -> App Settings
                        Container(
                          width: double.infinity,
                          margin: const EdgeInsets.only(bottom: 18),
                          decoration: BoxDecoration(
                            color: Colors.black54,
                            borderRadius:
                                BorderRadius.circular(_cardRadius + 6),
                          ),
                          child: ClipRRect(
                            borderRadius:
                                BorderRadius.circular(_cardRadius + 6),
                            child: BackdropFilter(
                              filter: ImageFilter.blur(sigmaX: 6, sigmaY: 6),
                              child: Padding(
                                padding: const EdgeInsets.all(20.0),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    const Text('Settings',
                                        style: TextStyle(
                                            fontSize: 20,
                                            fontWeight: FontWeight.bold)),
                                    const SizedBox(height: 12),
                                    // App Settings (thay cho General)
                                    _buildRowItem(
                                      icon: Icons.tune_rounded,
                                      title: 'App Settings',
                                      subtitle: 'Appearance & notifications',
                                      onTap: () {
                                        Navigator.push(
                                          context,
                                          MaterialPageRoute(
                                              builder: (_) =>
                                                  const GeneralSettingsScreen()),
                                        );
                                      },
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        ),

                        // Card: Extras (FAQ / Feedback)
                        Container(
                          width: double.infinity,
                          decoration: BoxDecoration(
                            color: Colors.black54,
                            borderRadius:
                                BorderRadius.circular(_cardRadius + 6),
                          ),
                          child: ClipRRect(
                            borderRadius:
                                BorderRadius.circular(_cardRadius + 6),
                            child: BackdropFilter(
                              filter: ImageFilter.blur(sigmaX: 6, sigmaY: 6),
                              child: Padding(
                                padding: const EdgeInsets.all(16.0),
                                child: Column(
                                  children: [
                                    _buildRowItem(
                                      icon: Icons.help_outline,
                                      title: 'FAQ',
                                      onTap: () {
                                        Navigator.push(
                                          context,
                                          MaterialPageRoute(
                                              builder: (_) =>
                                                  const FAQScreen()),
                                        );
                                      },
                                    ),
                                    const Divider(color: Colors.white12),
                                    _buildRowItem(
                                      icon: Icons.feedback_outlined,
                                      title: 'Feedback',
                                      onTap: () {
                                        Navigator.push(
                                          context,
                                          MaterialPageRoute(
                                              builder: (_) =>
                                                  const FeedbackScreen()),
                                        );
                                      },
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        ),

                        const SizedBox(height: 28),
                        // small footer text
                        Text('App version 1.0.0',
                            style: TextStyle(color: Colors.white38)),
                        const SizedBox(height: 36),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),

      // Bottom nav
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      floatingActionButton: const SizedBox.shrink(),

      bottomNavigationBar: SizedBox(
        height: 72 + bottomSafe,
        child: Container(
          padding: EdgeInsets.only(bottom: bottomSafe),
          decoration: BoxDecoration(
            color: const Color.fromARGB(230, 28, 16, 77),
            boxShadow: [
              BoxShadow(
                color: const Color.fromARGB(41, 55, 15, 156),
                offset: const Offset(0, -1),
                blurRadius: 4,
              ),
            ],
          ),
          child: ClipRRect(
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 2, sigmaY: 2),
              child: BottomNavigationBar(
                currentIndex: 2,
                onTap: (i) {
                  if (i == 0) {
                    Navigator.pushReplacementNamed(context, '/home');
                  } else if (i == 1) {
                    Navigator.pushReplacementNamed(context, '/sound');
                  } else if (i == 2) {
                    // already on Settings
                  }
                },
                backgroundColor: const Color.fromARGB(0, 67, 33, 152),
                elevation: 0,
                selectedItemColor: Colors.white,
                unselectedItemColor: Colors.white70,
                items: const [
                  BottomNavigationBarItem(
                    icon: Icon(Icons.alarm_rounded),
                    label: "Alarm",
                  ),
                  BottomNavigationBarItem(
                    icon: Icon(Icons.bar_chart_rounded),
                    label: "Sound",
                  ),
                  BottomNavigationBarItem(
                    icon: Icon(Icons.settings_rounded),
                    label: "Settings",
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
