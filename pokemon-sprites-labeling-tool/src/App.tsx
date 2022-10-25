import { ChangeEvent, useEffect, useState } from "react";
import * as Jimp from "jimp/browser/lib/jimp";
import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import { open } from "@tauri-apps/api/dialog";
import { invoke } from "@tauri-apps/api";

const cropProperties = {
  resizedW: 1280,
  resizedH: 720,
  beginW: 85,
  beginH: 20,
  boxW: 75,
  boxH: 75,
  offsetX: 588,
  offsetY: 186,
  boxCount: 6,
};

const cropBoxes = [
  { x: cropProperties.beginW, y: cropProperties.beginH },
  {
    x: cropProperties.beginW,
    y: cropProperties.beginH + cropProperties.offsetY,
  },
  {
    x: cropProperties.beginW,
    y: cropProperties.beginH + cropProperties.offsetY * 2,
  },
  {
    x: cropProperties.beginW + cropProperties.offsetX,
    y: cropProperties.beginH,
  },
  {
    x: cropProperties.beginW + cropProperties.offsetX,
    y: cropProperties.beginH + cropProperties.offsetY,
  },
  {
    x: cropProperties.beginW + cropProperties.offsetX,
    y: cropProperties.beginH + cropProperties.offsetY * 2,
  },
];

const readURL = (file: File) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target!.result);
    reader.onerror = (e) => reject(e);
    reader.readAsDataURL(file);
  });
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [imgURL, setImgURL] = useState<string | null>(null);
  const [croppedImages, setCroppedImages] = useState<string[]>([]);

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) {
      return;
    }
    const file = e.target.files[0];
    setFile(file);
  };

  useEffect(() => {
    if (file == null) return;
    readURL(file)
      .then((url) => {
        return Jimp.read(
          Buffer.from(
            (url as string).replace(/^data:image\/\w+;base64,/, ""),
            "base64"
          )
        );
      })
      .then((image) => {
        const resizedImage = image.resize(
          cropProperties.resizedW,
          cropProperties.resizedH
        );
        const croppedImages = cropBoxes
          .map((box) => {
            return resizedImage
              .clone()
              .crop(box.x, box.y, cropProperties.boxW, cropProperties.boxH);
          })
          .map((image) => {
            return image.getBase64Async(Jimp.MIME_PNG);
          });
        Promise.all(croppedImages).then((images) => {
          setCroppedImages(images);
        });
      })
      .catch((err) => {
        console.error(err);
      });
  }, [file]);

  return (
    <Container maxWidth="md">
      <h1>Upload images</h1>
      <Grid
        container
        spacing={{ xs: 2, md: 3 }}
        columns={{ xs: 4, sm: 8, md: 12 }}
      >
        {croppedImages.map((image, i) => (
          <Grid item xs={2} sm={4} md={4} key={i}>
            <img src={image} alt={`cropped image ${i}`} key={i} />
          </Grid>
        ))}
      </Grid>
      <div>
        <Button component="label" variant="outlined">
          Upload Image
          <input
            type="file"
            accept="image/*"
            hidden
            onChange={handleFileUpload}
          />
        </Button>
        <Button
          variant="contained"
          onClick={async () => {
            const selected = await open({
              multiple: true,
              filters: [
                {
                  name: "Image",
                  extensions: ["png", "jpeg", "jpg"],
                },
              ],
            });
            if (selected == null) return;
            const paths = Array.isArray(selected) ? selected : [selected];
            paths.forEach(async (path) => {
              invoke("open_resize_crop_image", { path: path }).then(
                (response) => {
                  setCroppedImages(response as string[]);
                }
              );
            });
          }}
        >
          Upload
        </Button>
      </div>
    </Container>
  );
}

export default App;
