export const PrivateContent = ({ children }: { children: React.ReactNode }) => {
  return (
    <div
      style={{
        width: "100%",
        minHeight: '100vh',
        height: "100%",
        display: "flex",
        flexDirection: 'column',
        backgroundColor: '#ffffff',
      }}
    >
      {children}
    </div>
  );
};
