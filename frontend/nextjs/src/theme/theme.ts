import { createTheme, ThemeOptions } from "@mui/material/styles";


export const themeOptions: ThemeOptions = {
  palette: {
    // type: 'light',
    primary: {
      main: '#28a745',
    },
    secondary: {
      main: '#007bff',
    },
    error: {
      main: '#d32f2f',
    },
    warning: {
      main: '#ffb74d',
    },
    background: {
      default: '#ffffff',
    },
  },
  typography: {
    fontFamily: 'Open Sans',
    h1: {
      fontFamily: 'Ubuntu Mono',
    },
    h2: {
      fontFamily: 'Ubuntu Mono',
    },
    h3: {
      fontFamily: 'Ubuntu Mono',
    },
    h4: {
      fontFamily: 'Ubuntu Mono',
    },
    h6: {
      fontFamily: 'Ubuntu Mono',
    },
    h5: {
      fontFamily: 'Ubuntu Mono',
    },
    subtitle1: {
      fontFamily: 'Ubuntu Mono',
    },
    subtitle2: {
      fontFamily: 'Ubuntu Mono',
    },
    button: {
      fontFamily: 'Ubuntu Mono',
      fontWeight: 900,
    },
    overline: {
      fontFamily: 'Ubuntu Mono',
    },
  },
};

export const theme = createTheme(
    themeOptions
//  palette: {
//    primary: {
//     //  main: "#fcba03",
//    },
//  },
);