plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.alarm_app"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        // dùng Java 1.8 để desugaring hoạt động ổn định
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8

        // Bật core library desugaring (Kotlin DSL)
        isCoreLibraryDesugaringEnabled = true
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    defaultConfig {
        applicationId = "com.example.alarm_app"
        // đảm bảo flutter.minSdkVersion >= 21
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    // nếu bạn có source set kotlin
    sourceSets {
        getByName("main") {
            java.srcDir("src/main/kotlin")
        }
    }
}

flutter {
    source = "../.."
}

// Thêm block dependencies nếu chưa có
dependencies {
    
    implementation("org.jetbrains.kotlin:kotlin-stdlib:1.8.22")

    // bắt buộc: desugar library
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.0.3")

    // các dependency khác của app nếu có
    implementation("androidx.core:core-ktx:1.10.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.0")
}
