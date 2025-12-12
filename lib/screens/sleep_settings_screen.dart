// lib/screens/sleep_settings_screen.dart
import 'package:flutter/material.dart';

class SleepSettingsScreen extends StatefulWidget {
  const SleepSettingsScreen({super.key});

  @override
  State<SleepSettingsScreen> createState() => _SleepSettingsScreenState();
}

class _SleepSettingsScreenState extends State<SleepSettingsScreen> {
  bool sleepTracking = true;
  bool smartWake = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Sleep Settings'),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: Container(
        padding: const EdgeInsets.all(18),
        decoration: const BoxDecoration(
          gradient: LinearGradient(
              colors: [Color(0xFF141328), Color(0xFF2A2140)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight),
        ),
        child: Column(
          children: [
            Card(
              color: Colors.white10,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
              child: Column(
                children: [
                  SwitchListTile(
                    title: const Text('Enable sleep tracking',
                        style: TextStyle(color: Colors.white)),
                    subtitle: const Text('Track sleep automatically',
                        style: TextStyle(color: Colors.white70)),
                    value: sleepTracking,
                    onChanged: (v) => setState(() => sleepTracking = v),
                    activeColor: Colors.blueAccent,
                  ),
                  const Divider(color: Colors.white12),
                  SwitchListTile(
                    title: const Text('Smart wake-up window',
                        style: TextStyle(color: Colors.white)),
                    subtitle: const Text(
                        'Wake during light sleep within window',
                        style: TextStyle(color: Colors.white70)),
                    value: smartWake,
                    onChanged: (v) => setState(() => smartWake = v),
                    activeColor: Colors.blueAccent,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),
            const Text(
              'These settings help the app to detect and analyze your sleep cycles. You can tune tracking sensitivity or allow integrations from wearables.',
              style: TextStyle(color: Colors.white70),
            ),
            const Spacer(),
            ElevatedButton(
              onPressed: () {
                ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Sleep settings saved')));
                Navigator.pop(context);
              },
              child: const Text('Save'),
            )
          ],
        ),
      ),
    );
  }
}
