import * as Jimp from "jimp/browser/lib/jimp";
import { ChangeEvent, useMemo, useState } from "react";
import toast, { Toaster } from "react-hot-toast";

import Autocomplete from "@mui/material/Autocomplete";
import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import FormControl from "@mui/material/FormControl";
import Grid from "@mui/material/Grid";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select, { SelectChangeEvent } from "@mui/material/Select";
import TextField from "@mui/material/TextField";
import { Icons } from "@pkmn/img";
import { invoke } from "@tauri-apps/api";

import { getDex } from "./utils/pkmn";

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
  speciesIds,
  onChange,
}: {
  value: string;
  speciesIds: string[];
  onChange: (event: ChangeEvent<{}>, newValue: string | null) => void;
}) {
  return (
    <Autocomplete
      value={value}
      onChange={onChange}
      options={speciesIds}
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

function GenSelect({
  genNumber,
  setGenNumber,
}: {
  genNumber: number;
  setGenNumber: (genNumber: number) => void;
}) {
  const handleChange = (event: SelectChangeEvent) => {
    setGenNumber(+event.target.value);
  };

  return (
    <FormControl sx={{ m: 1, minWidth: 80 }}>
      <InputLabel id="gen-select-label">Gen</InputLabel>
      <Select
        labelId="gen-select-label"
        id="gen-select"
        value={genNumber.toString()}
        label="Gen"
        autoWidth
        onChange={handleChange}
      >
        {[1, 2, 3, 4, 5, 6, 7, 8].map((g) => (
          <MenuItem key={g} value={g}>
            {g}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}

function App() {
  const [croppedImages, setCroppedImages] = useState<string[]>([]);
  const [species, setSpecies] = useState<string[]>([]);
  const [genNumber, setGenNumber] = useState<number>(8);

  const speciesIds = useMemo(
    () => Array.from(getDex(genNumber).species).map((s) => s.id),
    [genNumber]
  );

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
    const promise = invoke("write_images", {
      base64Images: croppedImages,
      paths: species.map((s) => `./out/${s}/${+new Date()}.png`),
    });
    toast.promise(promise, {
      loading: "Saving...",
      success: "Saved!",
      error: "Error on saving",
    });
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
              speciesIds={speciesIds}
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
        <GenSelect genNumber={genNumber} setGenNumber={setGenNumber} />
      </Grid>
      <Toaster />
    </Container>
  );
}

export default App;
