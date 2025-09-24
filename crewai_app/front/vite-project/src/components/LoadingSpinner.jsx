import styled, { keyframes } from 'styled-components';

const spin = keyframes`
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }

  50% {
    opacity: 0.5;
    transform: scale(1.5);
  }
`;

const SpinnerContainer = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 100vh;
  top: 0;
  left: 0;
  width: 100%;
  background-color: rgba(255, 255, 255, 0.8);
  z-index: 9999;
  position: fixed;
`;

const SpinnerWrapper = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
`;

const Spinner = styled.div`
  width: 15px;
  height: 15px;
  margin: 0 5px;
  background-color: #007bff;
  border-radius: 50%;
  animation: ${spin} 1.2s ease-in-out infinite;
`;

const LoadingText = styled.p`
  margin-top: 20px;
  font-size: 20px;
  color: #007bff;
  font-weight: bold;
  text-align: center;
  line-height: 1.5;
`;

const LoadingSpinner = () => {
  return (
    <SpinnerContainer>
      <SpinnerWrapper>
        <Spinner></Spinner>
        <Spinner></Spinner>
        <Spinner></Spinner>
      </SpinnerWrapper>
      <LoadingText>블로그 콘텐츠 생성중</LoadingText>
    </SpinnerContainer>
  );
};

export default LoadingSpinner;
