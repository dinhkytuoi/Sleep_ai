// lib/screens/sound_screen.dart
import 'dart:ui';
import 'package:flutter/material.dart';
import 'sound_detail_screen.dart';
import '../services/sound_manager.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SoundScreen extends StatefulWidget {
  const SoundScreen({super.key});

  @override
  State<SoundScreen> createState() => _SoundScreenState();
}

class _SoundScreenState extends State<SoundScreen> {
  final List<Map<String, String>> mixes = [
    {
      'title': 'Drizzling',
      'image': 'assets/images/dizzing.png',
      'audio': 'assets/audio/drizzling.mp3'
    },
    {
      'title': 'Raindrops Drum',
      'image': 'assets/images/rd.png',
      'audio': 'assets/audio/raindrops_drum.mp3'
    },
    {
      'title': 'Summer Rain',
      'image': 'assets/images/summer.png',
      'audio': 'assets/audio/summer_rain.mp3'
    },
    {
      'title': 'Stepping Rain',
      'image': 'assets/images/sr.png',
      'audio': 'assets/audio/stepping_rain.mp3'
    },
    {
      'title': 'Rain in Forest',
      'image': 'assets/images/rf.png',
      'audio': 'assets/audio/rain_forest.mp3'
    },
    {
      'title': 'Showers on Window',
      'image': 'assets/images/wr.png',
      'audio': 'assets/audio/showers_window.mp3'
    },
    {
      'title': 'Gentle Stream',
      'image': 'assets/images/stream.png',
      'audio': 'assets/audio/gentle_stream.mp3'
    },
    {
      'title': 'Night Thunder',
      'image': 'assets/images/thunder.png',
      'audio': 'assets/audio/night_thunder.mp3'
    },
  ];

  // chứa các asset audio đã được download (chỉ lưu key asset path)
  Set<String> downloadedSet = {};

  @override
  void initState() {
    super.initState();
    _loadDownloadedSet();
  }

  Future<void> _loadDownloadedSet() async {
    final map = await SoundManager.getAllDownloaded();
    if (!mounted) return;
    setState(() => downloadedSet = map.keys.toSet());
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final bottomSafe = MediaQuery.of(context).padding.bottom;

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
          child: Column(
            children: [
              // Header
              Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 20.0, vertical: 18),
                child: Row(
                  children: [
                    Expanded(
                      child: Text(
                        'Sleep Sounds',
                        style: theme.textTheme.headlineSmall?.copyWith(
                          fontWeight: FontWeight.bold,
                          fontSize: 32,
                          color: Colors.white,
                        ),
                      ),
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
              ),

              const SizedBox(height: 12),

              // Grid of mixes
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16.0),
                  child: GridView.builder(
                    padding: const EdgeInsets.only(bottom: 16),
                    itemCount: mixes.length,
                    gridDelegate:
                        const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 2,
                      mainAxisSpacing: 18,
                      crossAxisSpacing: 16,
                      childAspectRatio: 1,
                    ),
                    itemBuilder: (context, i) {
                      final item = mixes[i];
                      final audioAsset = item['audio'] ?? '';
                      final isDownloaded = downloadedSet.contains(audioAsset);

                      return GestureDetector(
                        onTap: () async {
                          // 🔹 Lưu âm thanh được chọn vào SharedPreferences
                          final prefs = await SharedPreferences.getInstance();
                          await prefs.setString(
                              'selectedSound', item['audio']!);

                          // 🔹 Mở màn chi tiết âm thanh
                          await Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => SoundDetailScreen(
                                title: item['title']!,
                                imageAsset: item['image']!,
                                audioAsset: item['audio']!,
                              ),
                            ),
                          );

                          // 🔹 Reload trạng thái download nếu có thay đổi
                          await _loadDownloadedSet();

                          // 🔹 Hiển thị thông báo đã chọn
                          if (context.mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content:
                                    Text('✅ Selected sound: ${item['title']}'),
                                duration: const Duration(seconds: 2),
                              ),
                            );
                          }
                        },
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(18),
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              Image.asset(
                                item['image']!,
                                fit: BoxFit.cover,
                                errorBuilder: (_, __, ___) => Container(
                                  color: Colors.black12,
                                  child: Column(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: const [
                                      Icon(Icons.image_not_supported,
                                          size: 40, color: Colors.white54),
                                      SizedBox(height: 8),
                                      Text(
                                        'No image',
                                        style: TextStyle(
                                            color: Colors.white54,
                                            fontSize: 12),
                                      ),
                                    ],
                                  ),
                                ),
                              ),

                              // overlay gradient để chữ dễ đọc
                              Container(
                                decoration: BoxDecoration(
                                  gradient: LinearGradient(
                                    colors: [
                                      Colors.black.withOpacity(0.45),
                                      Colors.black26
                                    ],
                                    begin: Alignment.bottomCenter,
                                    end: Alignment.topCenter,
                                  ),
                                ),
                              ),

                              // download/check icon
                              Positioned(
                                right: 8,
                                top: 8,
                                child: Container(
                                  decoration: BoxDecoration(
                                    color: Colors.black45,
                                    shape: BoxShape.circle,
                                  ),
                                  padding: const EdgeInsets.all(6),
                                  child: Icon(
                                    isDownloaded
                                        ? Icons.check_circle
                                        : Icons.download_rounded,
                                    size: 18,
                                    color: isDownloaded
                                        ? Colors.greenAccent
                                        : Colors.white,
                                  ),
                                ),
                              ),

                              // title
                              Positioned(
                                left: 12,
                                bottom: 12,
                                right: 12,
                                child: Text(
                                  item['title']!,
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 18,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ),
            ],
          ),
        ),
      ),

      // bottom nav
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
                currentIndex: 1,
                onTap: (i) {
                  if (i == 0) {
                    Navigator.pushReplacementNamed(context, '/home');
                  } else if (i == 1) {
                    // already on Sound
                  } else if (i == 2) {
                    Navigator.pushReplacementNamed(context, '/settings');
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
