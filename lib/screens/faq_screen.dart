// lib/screens/faq_screen.dart
import 'package:flutter/material.dart';

class FAQScreen extends StatelessWidget {
  const FAQScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final faqs = [
      {
        'q': 'How to enable sleep tracking?',
        'a': 'Go to Sleep Settings and enable tracking.'
      },
      {
        'q': 'How to change alarm tone?',
        'a': 'Open Alarm detail and edit Tone.'
      },
      {
        'q': 'Where are my saved sleeps?',
        'a': 'Statistics tab will show sleep summaries.'
      },
    ];

    return Scaffold(
      appBar: AppBar(
          title: const Text('FAQ'),
          backgroundColor: Colors.transparent,
          elevation: 0),
      body: Container(
        padding: const EdgeInsets.all(18),
        decoration: const BoxDecoration(
          gradient:
              LinearGradient(colors: [Color(0xFF141328), Color(0xFF2A2140)]),
        ),
        child: ListView.separated(
          itemBuilder: (ctx, i) {
            final item = faqs[i];
            return ExpansionTile(
              title:
                  Text(item['q']!, style: const TextStyle(color: Colors.white)),
              children: [
                Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Text(item['a']!,
                      style: const TextStyle(color: Colors.white70)),
                )
              ],
            );
          },
          separatorBuilder: (_, __) => const Divider(color: Colors.white12),
          itemCount: faqs.length,
        ),
      ),
    );
  }
}
