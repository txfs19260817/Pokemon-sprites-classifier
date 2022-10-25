#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]
extern crate base64;
extern crate image;

use base64::{decode, encode};
use std::io::Cursor;

const RESIZED_WIDTH: u32 = 1280;
const RESIZED_HEIGHT: u32 = 720;
const BEGIN_WIDTH: u32 = 85;
const BEGIN_HEIGHT: u32 = 20;
const BOX_WIDTH: u32 = 75;
const BOX_HEIGHT: u32 = 75;
const OFFSET_X: u32 = 588;
const OFFSET_Y: u32 = 186;
const BOXES_COUNT: u32 = 6;


#[tauri::command]
fn open_resize_crop_image(path: &str) -> Vec<String> {
    let image = image::open(path).unwrap();
    let resized = image::imageops::resize(
        &image,
        RESIZED_WIDTH,
        RESIZED_HEIGHT,
        image::imageops::FilterType::Lanczos3,
    );
    let mut cropped_images = Vec::new();
    for i in 0..BOXES_COUNT {
        let cropped = image::imageops::crop_imm(
            &resized,
            BEGIN_WIDTH + OFFSET_X * (i % 2),
            BEGIN_HEIGHT + OFFSET_Y * (i % 3),
            BOX_WIDTH,
            BOX_HEIGHT,
        )
        .to_image();
        cropped_images.push(cropped);
    }
    let mut base64_images: Vec<String> = Vec::new();
    for cropped in cropped_images {
        let mut buffer = vec![];
        cropped.write_to(&mut Cursor::new(&mut buffer), image::ImageOutputFormat::Png).expect("Failed to write image");
        let base64_image = encode(&buffer);
        base64_images.push(format!("data:image/png;base64,{}",base64_image));
    }
    base64_images
}

#[tauri::command]
fn write_image(path: &str, base64_image: &str) {
    let image = decode(base64_image).unwrap();
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&image).unwrap();
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![open_resize_crop_image, write_image])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
