import { Box, Button, IconButton, Typography } from "@/app/components/atoms/index";
import { DeleteOutline, ArrowUpward, ArrowDownward } from "@/assets/icons";
import { useEffect, useState } from "react";

type Props = {
  titleElement?: string;
  formId: string;
  data: any[];
  children: React.ReactElement[];
  onDataUpdate: (data: any[]) => void;
  addButtonTitle?: string;
};

export const MultiElementsContainer = (props: Props) => {
  const { children, data, onDataUpdate, addButtonTitle, formId, titleElement } = props;

  const [elementsData, setElementsData] = useState<any[]>([{}]);

  useEffect(() => {
    setElementsData(data);
  }, [data]);

  const onAddition = () => {
    setElementsData((elementsData) => {
      const updatedData = [...elementsData, {}];
      console.log("data", elementsData, updatedData);
      onDataUpdate(updatedData);
      return updatedData;
    });
  };

  const onRemove = (index: number) => () => {
    setElementsData((elementsData) => {
      const updatedData = [
        ...elementsData.slice(0, index),
        ...elementsData.slice(index + 1),
      ];
      console.log("data", elementsData, updatedData);

      const effectiveData =
        !updatedData || updatedData.length === 0 ? [{}] : updatedData;
      onDataUpdate(effectiveData);
      return effectiveData;
    });
  };

  const onPositionUp = (index: number) => {};

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        width: '100%'
      }}
    >
      {children?.map((element, index) => {
        return (
          <Box
            key={`multi-form-${formId}-${index}`}
            sx={{
              m: "10px",
              border: "1px ridge #ccc",
              background: "#cccccc30",
              width: '100%'
            }}
          >
            <Box
              sx={{
                p: "5px 10px",
                display: "flex",
                border: "1px ridge #ccc",
                justifyContent: 'space-between',
              }}
            >
              
              <Typography variant='h5'>{(titleElement && elementsData[index][titleElement])  ||  ''}&nbsp;</Typography>
              <Box>
              {elementsData.length > 1 && (
                <IconButton onClick={onRemove(index)}>
                  <DeleteOutline />
                </IconButton>
              )}

              {index > 0 && (
                <IconButton onClick={() => console.log("Position Up Pressed")}>
                  <ArrowUpward />
                </IconButton>
              )}
              {index < children.length - 1 && (
                <IconButton
                  onClick={() => console.log("Position Down Pressed")}
                >
                  <ArrowDownward />
                </IconButton>
              )}
              </Box>

            </Box>
            <Box
              sx={{
                p: "10px 20px",
                display: "flex",
                border: "1px ridge #ccc",
                justifyContent: "flex-end",
                width: '100%'
              }}
            >
              {element}
            </Box>
          </Box>
        );
      })}
      <Box
        sx={{
          m: "10px",
          p: "10px",
          width: "100%",
          border: "1px dotted #ccc",
          display: "flex",
          justifyContent: "center",
        }}
      >
        <Button onClick={onAddition} variant="outlined">
          {addButtonTitle ?? "Add"}
        </Button>
      </Box>
    </Box>
  );
};
