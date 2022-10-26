import { ChangeEvent, useState } from "react";
import * as Jimp from "jimp/browser/lib/jimp";
import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import { invoke } from "@tauri-apps/api";
import { dex } from "./utils/pkmn";
import Autocomplete from "@mui/material/Autocomplete";
import TextField from "@mui/material/TextField";
import { Icons } from "@pkmn/img";

const speciesId = Array.from(dex.species).map((s) => s.id);

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

const openBase64Image = async (base64: string) => {
  return Jimp.read(
    Buffer.from(base64.replace(/^data:image\/\w+;base64,/, ""), "base64")
  );
};

const resizeAndCrop = async (image: Jimp): Promise<string[]> => {
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
  return Promise.all(croppedImages);
};

function SpeciesAutocomplete({
  value,
  onChange,
}: {
  value: string;
  onChange: (event: ChangeEvent<{}>, newValue: string | null) => void;
}) {
  return (
    <Autocomplete
      value={value}
      onChange={onChange}
      options={speciesId}
      renderInput={(params) => <TextField {...params} label="Species" />}
      renderOption={(props, option) => (
        <li {...props}>
          <span style={Icons.getPokemon(option).css} />
          {option}
        </li>
      )}
      filterOptions={(options, params) =>
        options.filter((option) => option.startsWith(params.inputValue))
      }
    />
  );
}

function App() {
  const [croppedImages, setCroppedImages] = useState<string[]>([]);
  const [species, setSpecies] = useState<string[]>([]);

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) {
      return;
    }
    const croppedImages = (
      await Promise.all(
        Array.from(e.target.files).map(
          async (file) =>
            await readURL(file)
              .then((base64) => openBase64Image(base64 as string))
              .then((image) => resizeAndCrop(image))
              .then((images) => images)
        )
      )
    ).flat();
    setCroppedImages(croppedImages);
    setSpecies(croppedImages.map(() => "incineroar"));
  };

  const handleSave = async () => {
    const message = await invoke("write_images", {
      base64Images: croppedImages,
      paths: species.map(
        (s) => `./out/${s}/${Math.floor(Math.random() * 10000000)}.png`
      ),
    });
    alert(message);
  };

  return (
    <Container maxWidth="lg">
      <h1>Labeling Tool</h1>
      <Grid
        container
        spacing={{ xs: 2, md: 3 }}
        columns={{ xs: 4, sm: 8, md: 12 }}
      >
        {croppedImages.map((image, i) => (
          <Grid item xs={2} sm={4} md={4} key={i}>
            <img src={image} alt={`cropped image ${i}`} key={i} />
            <SpeciesAutocomplete
              value={species[i]}
              onChange={(_, v) => {
                species[i] = v as string;
                setSpecies([...species]);
              }}
            />
          </Grid>
        ))}
      </Grid>
      <Grid>
        <Button component="label" variant="outlined">
          Load Images
          <input
            type="file"
            multiple
            accept="image/*"
            hidden
            onChange={handleFileUpload}
          />
        </Button>
        <Button
          component="label"
          variant="contained"
          disabled={!species.length}
          onClick={handleSave}
        >
          Save
        </Button>
      </Grid>
    </Container>
  );
}

export default App;
