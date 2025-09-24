import './App.css';
import styled from 'styled-components';
import InputGroup from './components/InputGroup';
import LoadingSpinner from './components/LoadingSpinner';
import { useState } from 'react';

const AppContainer = styled.div`
  padding: 40px 20px;
  max-width: 800px;
  margin: 0 auto;
  font-family: 'Helvetica', sans-serif;
`;

const Header = styled.h1`
  text-align: center;
  color: #333;
  font-size: 2.5rem;
  margin-bottom: 20px;
`;

const Description = styled.p`
  text-align: center;
  margin-bottom: 40px;
  color: #666;
  font-size: 1.2rem;
`;

function App() {
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e) => {
    console.log(e.target.value);
  };
  return (
    <AppContainer>
      <Header>CrewAI 블로그 콘텐츠 생성기</Header>
      <Description>
        주제에 맞는 블로그 콘텐츠를 생성하기 위한 에이전트를 이용해 보세요.
      </Description>
      {loading ? (
        <LoadingSpinner />
      ) : (
        <InputGroup loading={loading} handleInputChange={handleInputChange} />
      )}
    </AppContainer>
  );
}

export default App;
